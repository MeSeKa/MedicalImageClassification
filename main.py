
# İLK DENEMEDİR SON KOD DEĞİLDİR. SON KOD SONU _different_approach olarak bitendir.


import os
import glob
import re
import random
import math
from collections import defaultdict
from utils import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

directory = "C:\\Users\\MehmedSefa\\GitHub\\MedicalImageClassification\\Dataset"


def create_balanced_splits(train_dir, test_dir, validation_ratio=0.2, random_seed=42):

    if not os.path.isdir(train_dir):
        print(f"Hata: Eğitim dizini bulunamadı: {train_dir}")
        return [], [], []
    if not os.path.isdir(test_dir):
        print(f"Hata: Test dizini bulunamadı: {test_dir}")
        return [], [], []

    random.seed(random_seed)  # Rastgeleliği sabitlemek için

    # --- Eğitim Verisini İşleme ---
    all_train_files = glob.glob(os.path.join(train_dir, "*.png"))
    if not all_train_files:
        print(
            f"Uyarı: Eğitim dizininde ('{train_dir}') hiç PNG dosyası bulunamadı.")
        # Eğitim ve validasyon boş dönecek

    # Dosyaları sınıflarına göre gruplandırma
    class_files = defaultdict(list)
    total_valid_train_samples = 0
    print("Eğitim dosyaları taranıyor ve sınıflar ayrıştırılıyor...")
    for filepath in all_train_files:
        class_label = parse_image_filename(filepath)
        if class_label:
            class_files[class_label].append(filepath)
            total_valid_train_samples += 1
        # else: Hata mesajı parse_image_filename içinde zaten veriliyor

    num_classes = len(class_files)
    if total_valid_train_samples == 0:
        print(
            "Uyarı: Eğitim dizininde geçerli isimlendirmeye sahip hiç dosya bulunamadı."
        )
        # Eğitim ve validasyon boş dönecek
    elif num_classes < 4:
        print(
            f"Uyarı: Beklenen 4 sınıf yerine sadece {num_classes} sınıf bulundu: {list(class_files.keys())}"
        )

    train_paths = []
    validation_paths = []

    if num_classes > 0 and total_valid_train_samples > 0:
        print(
            f"\nToplam {total_valid_train_samples} geçerli eğitim örneği bulundu.")
        print("Sınıf dağılımları (Eğitim başlangıç):")
        for label, files in class_files.items():
            print(f"  Sınıf '{label}': {len(files)} örnek")

        # Hedeflenen toplam validasyon örneği sayısı
        target_total_validation_samples = math.floor(
            total_valid_train_samples * validation_ratio
        )

        # Her sınıftan alınacak *eşit* validasyon örneği sayısı
        # Bu sayı, hedeflenen toplamın sınıflara bölünmesiyle bulunur,
        # ancak en küçük sınıfın boyutunu geçemez.
        num_val_per_class = (
            math.floor(target_total_validation_samples / num_classes)
            if num_classes > 0
            else 0
        )

        # En küçük sınıftaki örnek sayısını bul
        min_class_size = (
            min(len(paths)
                for paths in class_files.values()) if class_files else 0
        )

        # Eğer hesaplanan val sayısı en küçük sınıftan fazlaysa, en küçük sınıfa göre ayarla
        if num_val_per_class > min_class_size:
            print(
                f"\nUyarı: İstenen validasyon örnek sayısı/sınıf ({num_val_per_class}), en küçük sınıf boyutundan ({min_class_size}) büyük."
            )
            num_val_per_class = min_class_size
            print(
                f"Validasyon örnek sayısı/sınıf {num_val_per_class} olarak ayarlandı."
            )

        if num_val_per_class == 0 and validation_ratio > 0:
            print(
                "\nUyarı: Hesaplanan validasyon örnek sayısı/sınıf 0. Validasyon seti boş olacak."
            )
            print(
                "Bunun nedeni çok az toplam örnek, çok sayıda sınıf veya çok küçük bir sınıf olabilir."
            )

        print(
            f"\nValidasyon seti için her sınıftan {num_val_per_class} örnek alınacak."
        )

        # Her sınıfı ayrı ayrı böl ve listelere ekle
        for class_label, paths in class_files.items():
            random.shuffle(paths)  # Sınıf içindeki dosyaları karıştır

            # Bu sınıftan alınacak gerçek validasyon sayısı (eğer sınıf küçükse num_val_per_class'tan az olabilir)
            actual_val_count = min(num_val_per_class, len(paths))

            validation_paths.extend(paths[:actual_val_count])
            train_paths.extend(paths[actual_val_count:])

        # Son listeleri de karıştır
        random.shuffle(train_paths)
        random.shuffle(validation_paths)

    # --- Test Verisini İşleme ---
    print("\nTest dosyaları taranıyor...")
    test_paths = glob.glob(os.path.join(test_dir, "*.png"))
    if not test_paths:
        print(
            f"Uyarı: Test dizininde ('{test_dir}') hiç PNG dosyası bulunamadı.")

    # --- Sonuçları Yazdırma ---
    print("\n--- Veri Seti Bölme Özeti ---")
    print(f"Toplam Geçerli Eğitim Örneği: {total_valid_train_samples}")
    print(f"Oluşturulan Eğitim Seti Boyutu: {len(train_paths)}")
    print(
        f"Oluşturulan Validasyon Seti Boyutu: {len(validation_paths)} (Dengeli)")
    print(f"Bulunan Test Seti Boyutu: {len(test_paths)}")

    # Validasyon setinin dengesini kontrol et (opsiyonel ama faydalı)
    if validation_paths:
        val_class_counts = defaultdict(int)
        for f in validation_paths:
            class_label = parse_image_filename(f)
            if class_label:
                val_class_counts[class_label] += 1
        print(f"Validasyon Seti Sınıf Dağılımı: {dict(val_class_counts)}")

    return train_paths, validation_paths, test_paths


TRAIN_DATA_DIR = directory + "\\train"
TEST_DATA_DIR = directory + "\\test"


train_files, val_files, test_files = create_balanced_splits(
    train_dir=TRAIN_DATA_DIR,
    test_dir=TEST_DATA_DIR,
    validation_ratio=0.20,  # %20 validasyon seti için
    random_seed=42,  # Tekrarlanabilirlik için sabit bir sayı
)


class MedicalImageDataset(Dataset):
    def __init__(self, imagePaths, transform=None):
        self.transform = transform
        self.classes = ["0", "1+", "2+", "3+"]
        self.class_to_idx = {cls_name: i for i,
                             cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        for img_path in imagePaths:
            #     string_label = parse_image_filename(img_path)
            #     numeric_label = self.class_to_idx.get(string_label)
            self.image_paths.append(img_path)
        #     self.labels.append(numeric_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        string_label = parse_image_filename(img_path)
        numeric_label = self.class_to_idx.get(string_label)
        label = numeric_label

        if self.transform:
            image = self.transform(image)

        return image, label


data_transform = transforms.Compose([
    # Zaten boyutlar 1024x1024 ama tedbir için
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = MedicalImageDataset(
    imagePaths=train_files, transform=data_transform)
val_dataset = MedicalImageDataset(
    imagePaths=val_files, transform=data_transform)
test_dataset = MedicalImageDataset(
    imagePaths=test_files, transform=data_transform)


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Sınıf adını giriş boyutunu belirtecek şekilde değiştirdim
class ModifiedMedicalCNN_1024Input(nn.Module):
    def __init__(self, num_classes=4):
        super(ModifiedMedicalCNN_1024Input, self).__init__()

        # --- Evrişim Katmanları (Convolutional Layers) ---
        # İstek 1: 4 katman, çıkışlar 8, 16, 32, 64
        # İstek 2: Aktivasyon (ReLU) ve Pooling (MaxPool2d) seçildi
        # Padding=1 kullanılarak Conv sonrası boyut korunuyor, Pool ile yarıya iniyor.

        # Blok 1
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1024 -> 512

        # Blok 2
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512 -> 256

        # Blok 3
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256 -> 128

        # Blok 4
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128 -> 64

        # --- Sınıflandırıcı Katmanları (Fully Connected Layers) ---

        # 1024x1024 GİRİŞ İÇİN HESAPLAMA:
        # Son pooling sonrası boyut: 64 kanal, 64x64 (H/16, W/16)
        # Düzleştirilmiş (flattened) boyut: 64 (kanal) * 64 (yükseklik) * 64 (genişlik) = 262144
        self.flattened_features = 64 * 64 * 64  # 262144

        # İstek 3: 3 ara katman (64, 128, 256 nöron) + çıkış katmanı
        # Giriş boyutu güncellendi
        self.fc1 = nn.Linear(
            in_features=self.flattened_features, out_features=64)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(in_features=128, out_features=256)
        self.relu_fc3 = nn.ReLU()

        # Çıkış Katmanı (Output Layer) - 4 sınıf için
        self.fc_out = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        # Input shape: (batch_size, 3, 1024, 1024)

        # Evrişim Blokları
        x = self.pool1(self.relu1(self.conv1(x)))  # -> (N, 8, 512, 512)
        x = self.pool2(self.relu2(self.conv2(x)))  # -> (N, 16, 256, 256)
        x = self.pool3(self.relu3(self.conv3(x)))  # -> (N, 32, 128, 128)
        x = self.pool4(self.relu4(self.conv4(x)))  # -> (N, 64, 64, 64)

        # Düzleştirme (Flatten)
        x = torch.flatten(x, 1)  # -> (N, 262144)

        # Sınıflandırıcı Katmanları
        x = self.dropout1(self.relu_fc1(self.fc1(x)))  # -> (N, 64)
        x = self.dropout2(self.relu_fc2(self.fc2(x)))  # -> (N, 128)
        x = self.relu_fc3(self.fc3(x))                # -> (N, 256)

        # Çıkış Katmanı
        x = self.fc_out(x)                            # -> (N, num_classes)

        return x


if __name__ == '__main__':

    model = ModifiedMedicalCNN_1024Input(num_classes=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop with validation
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
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
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate metrics
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print(
            f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       'best_modell_batchsize_32_epoch100.pth')
            print('Model saved!')

    print('Training complete!')
