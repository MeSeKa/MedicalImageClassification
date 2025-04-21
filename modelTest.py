import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from main_class_weighting import MedicalImageDataset, ModifiedMedicalCNN_1024Input

# Confusion Matrix ve çizim için importlar
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

directory = "C:\\Users\\MehmedSefa\\GitHub\\MedicalImageClassification\\Dataset"

TEST_DATA_DIR = os.path.join(directory, "test")
MODEL_PATH = "best_model_weighted_loss_stratified.pth"
CLASS_NAMES = ['0', '1+', '2+', '3+']  # Sınıf isimler,
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 32

test_files = []
for root, dirs, files in os.walk(TEST_DATA_DIR):
    for file in files:
        if file.endswith(".png"):
            test_files.append(os.path.join(root, file))

data_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset ve DataLoader
test_dataset = MedicalImageDataset(
    imagePaths=test_files, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- Model Yükleme ---
model = ModifiedMedicalCNN_1024Input(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()  # Değerlendirme modu


# --- Test Döngüsü ---
criterion = nn.CrossEntropyLoss()  # Kayıp hesaplamak için (opsiyonel ama faydalı)
test_loss = 0.0  # test_loss olarak değiştirdim
correct = 0
total = 0
all_labels = []
all_preds = []

print("\nTest süreci başlıyor...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()  # Sadece loss değerini topla

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Confusion matrix için etiketleri ve tahminleri topla
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# --- Metrik Hesaplama ve Yazdırma (Döngüden Sonra) ---
# Döngü bittikten sonra ortalamaları hesapla
# Toplam batch sayısına böl
avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
test_acc = correct / total if total > 0 else 0

print("\n--- Test Sonuçları ---")
print(f'Ortalama Test Kaybı (Loss): {avg_test_loss:.4f}')
print(f'Test Doğruluğu (Accuracy): {test_acc:.4f} ({correct}/{total})')


# --- Confusion Matrix Hesaplama ve Çizdirme ---
print("\nConfusion Matrix oluşturuluyor...")
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')

# Başlığı doğruluk ve kayıp değerleriyle güncelle
title = f'Accuracy: {test_acc:.4f}, Loss: {avg_test_loss:.4f}'
plt.title(title)

plt.savefig(MODEL_PATH.replace('.pth', '_confusion_matrix.png'))
plt.show()

print("\nTest tamamlandı.")
