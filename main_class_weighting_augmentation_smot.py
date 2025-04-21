# Gerekli yeni importlar
from sklearn.model_selection import train_test_split
# SMOTE yerine bunu kullanacağız
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import numpy as np

import os
import glob
import re
import random
import math
from collections import defaultdict
from utils import parse_image_filename
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import torchvision  # add_graph için gerekebilir

# Eğitim döngüsünden hemen önce
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir_base = "runs"  # Ana log dizini
experiment_name = f"oversampled_augmented_smot_{timestamp}"  # Deney adı
log_dir = os.path.join(log_dir_base, experiment_name)

writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard logları şuraya kaydedilecek: {log_dir}")


# --- Parametreler ve Yollar ---
directory = "C:\\Users\\MehmedSefa\\GitHub\\MedicalImageClassification\\Dataset"
TRAIN_DATA_DIR = os.path.join(directory, "train")
TEST_DATA_DIR = os.path.join(directory, "test")
random_seed = 42
validation_ratio = 0.20
batch_size = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

# --- 1. Stratified Train/Validation Split (Değişiklik yok) ---
print("Eğitim dosyaları taranıyor ve etiketler çıkarılıyor...")
all_train_files = glob.glob(os.path.join(TRAIN_DATA_DIR, "*.png"))
all_train_labels_str = [parse_image_filename(f) for f in all_train_files]

valid_files_labels = [(f, l) for f, l in zip(
    all_train_files, all_train_labels_str) if l is not None]
if not valid_files_labels:
    raise ValueError(
        f"Eğitim dizininde ('{TRAIN_DATA_DIR}') geçerli etiketli PNG dosyası bulunamadı.")

all_train_files = [item[0] for item in valid_files_labels]
all_train_labels_str = [item[1]
                        for item in valid_files_labels]  # String etiketler

print(f"Toplam {len(all_train_files)} geçerli eğitim örneği bulundu.")
print("Başlangıç Sınıf Dağılımı (Train):", Counter(all_train_labels_str))

# Stratified split yap
train_files, val_files, train_labels_str, val_labels_str = train_test_split(
    all_train_files,
    all_train_labels_str,
    test_size=validation_ratio,
    stratify=all_train_labels_str,
    random_state=random_seed
)

test_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.png"))

print("\n--- Veri Seti Bölme Özeti ---")
print(f"Bölme Sonrası Eğitim Seti Boyutu: {len(train_files)}")
print(f"Bölme Sonrası Validasyon Seti Boyutu: {len(val_files)}")
print(f"Bulunan Test Seti Boyutu: {len(test_files)}")
print("Bölme Sonrası Eğitim Seti Sınıf Dağılımı:", Counter(train_labels_str))
# print("Validasyon Seti Sınıf Dağılımı:", Counter(val_labels_str)) # İsteğe bağlı


# --- 2. RandomOverSampler Uygulaması (YENİ ADIM) ---
print("\nRandomOverSampler uygulanıyor (Azınlık sınıfları kopyalanacak)...")
# RandomOverSampler'ın çalışması için sayısal etiketlere ve placeholder 'X'e ihtiyacı var
# Önce sınıf->indeks map'ini oluşturalım
temp_classes = sorted(list(set(all_train_labels_str)))  # Tüm sınıfları al
class_to_idx = {cls_name: i for i, cls_name in enumerate(temp_classes)}
idx_to_class = {i: cls_name for cls_name, i in class_to_idx.items()}

# Eğitim etiketlerini sayısala çevir
train_labels_num = np.array([class_to_idx[label]
                            for label in train_labels_str])

# RandomOverSampler 'X' verisi bekler, dosya indislerini placeholder olarak kullanabiliriz
train_indices = np.arange(len(train_files)).reshape(-1, 1)

ros = RandomOverSampler(random_state=random_seed)
resampled_indices, resampled_labels_num = ros.fit_resample(
    train_indices, train_labels_num)

# Orijinal dosya listesinden yeni (oversampled) listeyi oluştur
resampled_train_files = [train_files[i] for i in resampled_indices.flatten()]
# İsteğe bağlı: Yeni etiket listesini de string'e çevirebiliriz
# resampled_train_labels_str = [idx_to_class[label] for label in resampled_labels_num]

print(
    f"Oversampling Sonrası Yeni Eğitim Seti Boyutu: {len(resampled_train_files)}")
print("Oversampling Sonrası Eğitim Seti Sınıf Dağılımı:", Counter(
    resampled_labels_num))  # Sayısal etiketlerle gösterim


# --- 3. Dataset ve DataLoader ---
class MedicalImageDataset(Dataset):
    # Dataset sınıfında değişiklik yapmaya gerek yok, sadece kullanılan dosya listesi değişecek
    def __init__(self, imagePaths, class_to_idx, transform=None):
        self.transform = transform
        self.class_to_idx = class_to_idx  # Dışarıdan al
        self.image_paths = imagePaths
        # Etiketleri burada önceden almak yerine __getitem__ içinde alalım
        # self.labels = [self.class_to_idx.get(parse_image_filename(p)) for p in imagePaths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Hata: Görüntü okunamadı {img_path} - {e}")
            # Hatalı durumda ne yapılacağına karar verin: dummy data, exception vs.
            # Şimdilik dummy bir tensor döndürelim ve ilk sınıf etiketini verelim
            return torch.zeros((3, 1024, 1024)), 0

        string_label = parse_image_filename(img_path)
        numeric_label = self.class_to_idx.get(string_label)

        if numeric_label is None:
            print(
                f"Hata: Geçersiz etiketli dosya Dataloader'a geldi: {img_path}")
            numeric_label = 0  # Veya raise Exception(...)

        if self.transform:
            image = self.transform(image)

        return image, numeric_label


# Veri Dönüşümleri (Augmentation içeren train ve içermeyen val/test)
train_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset'leri oluştur (train_dataset artık resampled listeyi kullanıyor)
# class_to_idx map'ini Dataset'e veriyoruz
train_dataset = MedicalImageDataset(
    imagePaths=resampled_train_files, class_to_idx=class_to_idx, transform=train_transform)
val_dataset = MedicalImageDataset(
    imagePaths=val_files, class_to_idx=class_to_idx, transform=val_test_transform)
test_dataset = MedicalImageDataset(
    imagePaths=test_files, class_to_idx=class_to_idx, transform=val_test_transform)

# DataLoader'ları oluştur
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle önemli!
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# --- 4. Model Tanımı (Değişiklik yok) ---
class ModifiedMedicalCNN_1024Input(nn.Module):
    # ... (Önceki kodla aynı, buraya tekrar kopyalamadım) ...
    def __init__(self, num_classes=4):  # num_classes'ı dinamik alalım
        super(ModifiedMedicalCNN_1024Input, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1024 -> 512
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512 -> 256
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256 -> 128
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128 -> 64
        self.flattened_features = 64 * 64 * 64  # 262144
        self.fc1 = nn.Linear(
            in_features=self.flattened_features, out_features=64)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=128, out_features=256)
        self.relu_fc3 = nn.ReLU()
        self.fc_out = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.dropout1(self.relu_fc1(self.fc1(x)))
        x = self.dropout2(self.relu_fc2(self.fc2(x)))
        x = self.relu_fc3(self.fc3(x))
        x = self.fc_out(x)
        return x


# --- 5. Eğitim Döngüsü (Loss fonksiyonundan ağırlık kaldırıldı) ---
if __name__ == '__main__':

    # Sınıf sayısını dinamik olarak alalım
    num_classes = len(idx_to_class)
    model = ModifiedMedicalCNN_1024Input(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCihaz kullanılıyor: {device}")
    model.to(device)

    # ----- SINIF AĞIRLIKLANDIRMA KALDIRILDI -----
    # RandomOverSampler kullandığımız için eğitim seti artık (yaklaşık olarak) dengeli.
    # Bu durumda loss fonksiyonuna ayrıca ağırlık vermek genellikle gereksizdir
    # ve bazen performansı düşürebilir. Ağırlıksız CrossEntropyLoss kullanıyoruz.
    criterion = nn.CrossEntropyLoss()  # weight parametresi yok!
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Eğitim Başlıyor (RandomOverSampler + Augmentation ile) ---")
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        # train_loader artık oversampled ve augment edilmiş veri sağlıyor
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Ağırlıksız loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        # Validation phase (Değişiklik yok, val_loader orijinal veriyi kullanıyor)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # Ağırlıksız loss
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate metrics
        # Dikkat: train_loss'u hesaplarken oversampled dataset boyutunu kullanmalıyız
        # len(train_dataset) olacak
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)   # len(val_dataset)
        val_acc = correct / total

        print(
            f'Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)

        for name, param in model.named_parameters():
            if param.requires_grad:  # Sadece eğitilebilir parametreler
                writer.add_histogram(f'Weights/{name}', param.data, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = 'best_model_oversampled_augmented_smot.pth'  # Yeni isim
            torch.save(model.state_dict(), save_path)
            print(f'Model kaydedildi: {save_path}')

    print('\nEğitim tamamlandı!')
