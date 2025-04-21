import os
import glob
import re
import random
import math
from collections import defaultdict, Counter
from utils import parse_image_filename
from sklearn.model_selection import train_test_split  # Stratified split için
import numpy as np  # Ağırlık hesaplaması için

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import torchvision

# Eğitim döngüsünden hemen önce
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir_base = "runs"  # Ana log dizini
experiment_name = f"oversampled_augmented_{timestamp}"  # Deney adı
log_dir = os.path.join(log_dir_base, experiment_name)

writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard logları şuraya kaydedilecek: {log_dir}")


# --- Veri Yolları ---
directory = "C:\\Users\\MehmedSefa\\GitHub\\MedicalImageClassification\\Dataset"
TRAIN_DATA_DIR = os.path.join(directory, "train")
TEST_DATA_DIR = os.path.join(directory, "test")
random_seed = 42
validation_ratio = 0.20
batch_size = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

# --- 1. Stratified Train/Validation Split ---
print("Eğitim dosyaları taranıyor ve etiketler çıkarılıyor...")
all_train_files = glob.glob(os.path.join(TRAIN_DATA_DIR, "*.png"))
all_train_labels = [parse_image_filename(f) for f in all_train_files]

# Geçerli etiketlere sahip dosyaları filtrele
valid_files_labels = [(f, l) for f, l in zip(
    all_train_files, all_train_labels) if l is not None]
if not valid_files_labels:
    raise ValueError(
        f"Eğitim dizininde ('{TRAIN_DATA_DIR}') geçerli etiketli PNG dosyası bulunamadı.")

all_train_files = [item[0] for item in valid_files_labels]
all_train_labels = [item[1] for item in valid_files_labels]

print(f"Toplam {len(all_train_files)} geçerli eğitim örneği bulundu.")
print("Başlangıç Sınıf Dağılımı (Train):", Counter(all_train_labels))

# Stratified split yap temel olay burada
train_files, val_files, train_labels, val_labels = train_test_split(
    all_train_files,
    all_train_labels,
    test_size=validation_ratio,
    stratify=all_train_labels,  # Sınıf oranlarını koru
    random_state=random_seed
)

# Test dosyalarını bul
test_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.png"))

print("\n--- Veri Seti Bölme Özeti ---")
print(f"Oluşturulan Eğitim Seti Boyutu: {len(train_files)}")
print(f"Oluşturulan Validasyon Seti Boyutu: {len(val_files)}")
print(f"Bulunan Test Seti Boyutu: {len(test_files)}")
print("Eğitim Seti Sınıf Dağılımı:", Counter(train_labels))
print("Validasyon Seti Sınıf Dağılımı:", Counter(val_labels))


# --- 2. Sınıf Ağırlıklarını Hesaplama (Class Weights) ---
# Sadece eğitim setindeki dağılıma göre hesapla!
class_counts = Counter(train_labels)
total_samples = len(train_labels)
num_classes = len(class_counts)

# Ağırlıklar: total_samples / (num_classes * count_per_class)
class_weights = {}
# Önce class_to_idx'i burada tanımla
# Alfabetik veya mantıksal sıra önemli
temp_classes = sorted(class_counts.keys())
class_to_idx = {cls_name: i for i, cls_name in enumerate(temp_classes)}

weights_list = [0.0] * num_classes
for class_name, count in class_counts.items():
    idx = class_to_idx[class_name]
    weight = total_samples / (num_classes * count)
    weights_list[idx] = weight
    class_weights[class_name] = weight  # İsteğe bağlı, yazdırmak için

class_weights_tensor = torch.tensor(weights_list, dtype=torch.float)

print("\n--- Sınıf Ağırlıkları ---")
print("Hesaplanan Ağırlıklar:", class_weights)
print("Tensor:", class_weights_tensor)


# --- 3. Dataset ve DataLoader ---
class MedicalImageDataset(Dataset):
    def __init__(self, imagePaths, transform=None):
        self.transform = transform
        # Sınıf sırasının ağırlıklarla eşleştiğinden emin ol
        self.classes = sorted(list(set(parse_image_filename(
            p) for p in imagePaths if parse_image_filename(p) is not None)))
        self.class_to_idx = {cls_name: i for i,
                             cls_name in enumerate(self.classes)}
        # Ağırlıklar için sıra kontrolü
        self.idx_to_class = {i: cls_name for cls_name,
                             i in self.class_to_idx.items()}

        self.image_paths = imagePaths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        string_label = parse_image_filename(img_path)
        numeric_label = self.class_to_idx.get(
            string_label)  # None dönebilir, handle etmeli

        if numeric_label is None:  # Gereksiz ama kontrol edelim ne olur ne olamz
            print(
                f"Hata: Geçersiz etiketli dosya Dataloader'a geldi: {img_path}")
            numeric_label = 0

        label = numeric_label

        if self.transform:
            image = self.transform(image)

        return image, label


# EĞİTİM için veri artırma içeren transform
train_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.RandomHorizontalFlip(p=0.5),  # %50 olasılıkla yatay çevir
    transforms.RandomRotation(degrees=15),  # +/- 15 derece rastgele döndür
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.1),  # Renk/parlaklık oynamaları
    # İhtiyaca göre başka augmentation'lar eklenebilir
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# VALİDASYON ve TEST için veri artırma İÇERMEYEN transform
val_test_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    # Burada augmentation YOK!
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset'leri oluştururken doğru transformları kullanın:
train_dataset = MedicalImageDataset(
    imagePaths=train_files, transform=train_transform)  # Augmentation'lı
val_dataset = MedicalImageDataset(
    imagePaths=val_files, transform=val_test_transform)   # Augmentation'sız
test_dataset = MedicalImageDataset(
    imagePaths=test_files, transform=val_test_transform)  # Augmentation'sız

# DataLoader'ları oluştur (Değişiklik yok)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# --- 4. Model Tanımı (Değişiklik yok) ---
class ModifiedMedicalCNN_1024Input(nn.Module):
    def __init__(self, num_classes=4):
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


# --- 5. Eğitim Döngüsü ---
if __name__ == '__main__':

    model = ModifiedMedicalCNN_1024Input(
        num_classes=len(train_dataset.classes))  # Dinamik al
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCihaz kullanılıyor: {device}")
    model.to(device)

    # Ağırlık tensorunu GPU'ya gönder
    class_weights_tensor = class_weights_tensor.to(device)

    # CrossEntropyLoss'u ağırlıklarla tanımla
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Eğitim Başlıyor ---")
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # Ağırlıklı loss hesaplanacak
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate metrics
        train_loss = train_loss / len(train_loader.dataset)
        # Ağırlıksız ortalama loss
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        print(
            f'Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)

        # Save best model (Val Loss'a göre)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = 'best_model_weighted_loss_stratified_augmentation.pth'
            torch.save(model.state_dict(), save_path)
            print(f'Model kaydedildi: {save_path}')

    print('\nEğitim tamamlandı!')
