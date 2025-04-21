from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split  # Stratified split için
from datetime import datetime
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Scheduler eklendi
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np  # Ağırlık hesaplaması için
import os
import glob
import random
import math
from collections import defaultdict, Counter
import seaborn as sns  # For heatmap visualization

from utils import parse_image_filename

# --- Sabitler ve Yapılandırma ---
# Ana veri seti yolu
DIRECTORY = "C:\\Users\\MehmedSefa\\GitHub\\MedicalImageClassification\\Dataset"
TRAIN_DATA_DIR = os.path.join(DIRECTORY, "train")
TEST_DATA_DIR = os.path.join(DIRECTORY, "test")

# Hiperparametreler
RANDOM_SEED = 42
VALIDATION_RATIO = 0.20
BATCH_SIZE = 256
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.3
INPUT_SIZE = 256


# --- Dataset Sınıf Tanımı
class MedicalImageDataset(Dataset):
    def __init__(self, imagePaths, class_to_idx, transform=None):
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.image_paths = []
        self.labels = []

        for p in imagePaths:
            string_label = parse_image_filename(p)
            if string_label is None:
                print(
                    f"Uyarı: Geçersiz dosya adı formatı, etiket alınamadı: {p}")
                continue  # Etiket alınamayan dosyayı atla

            numeric_label = self.class_to_idx.get(string_label)
            if numeric_label is not None:  # ne olur ne olmaz
                self.labels.append(numeric_label)
                # Sadece geçerli etiketi olan ve class_to_idx'te bulunanları ekle
                self.image_paths.append(p)
        if not self.image_paths:
            print(
                f"Uyarı: Dataset için geçerli hiçbir görüntü bulunamadı. Yollar: {imagePaths}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            # Görüntüyü aç ve RGB'ye dönüştür
            image = Image.open(img_path).convert("RGB")
            # Transform uygula
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            print(f"Hata: Görüntü dosyası bulunamadı: {img_path}")
            # Hata durumunda boş bir tensor döndür (veya None ve DataLoader'da ele al)
            image = torch.zeros((3, INPUT_SIZE, INPUT_SIZE))
            label = -1  # Hatalı etiket olarak işaretle gerek yok ama aşırı robust yazmak istiyoruz
        except Exception as e:
            print(
                f"Hata: Görüntü yüklenemedi veya dönüştürülemedi: {img_path} - {e}")
            image = torch.zeros((3, INPUT_SIZE, INPUT_SIZE))
            label = -1  # Hatalı etiket olarak işaretle gerek yok ama aşırı robust yazmak istiyoruz

        # Etiketi tensor olarak döndür
        return image, torch.tensor(label, dtype=torch.long)


# --- Model Tanımı ---
class ModifiedMedicalCNN(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super(ModifiedMedicalCNN, self).__init__()
        # Convolutional katmanlar
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256 -> 128
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128 -> 64
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64 -> 32
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32 -> 16

        # Aktivasyon
        self.relu = nn.ReLU()

        # Flatten işleminden sonraki özellik sayısını dinamik hesapla ROBUST KOD
        final_conv_size = INPUT_SIZE // 16
        self.flattened_features = 64 * final_conv_size * final_conv_size

        # Fully connected katmanlar
        self.fc1 = nn.Linear(
            in_features=self.flattened_features, out_features=64)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(in_features=128, out_features=256)
        self.fc_out = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        # Convolutional bloklar
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))

        # Flatten
        x = torch.flatten(x, 1)  # Batch boyutunu koru (start_dim=1)

        # Fully connected bloklar
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))  # Son FC öncesi ReLU
        # Çıkış katmanı (aktivasyon yok, CrossEntropyLoss kendisi halleder)
        x = self.fc_out(x)
        return x


# --- Ana Çalıştırma Bloğu ---
if __name__ == '__main__':

    # Tekrarlanabilirlik için rastgele tohumları ayarla
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # --- TensorBoard Kurulumu --- (main bloğu içinde)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir_base = "runs"
    experiment_name = f"different_approach_{timestamp}"
    log_dir = os.path.join(log_dir_base, experiment_name)
    # Log dizinini oluştur (varsa hata verme)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logları şuraya kaydedilecek: {log_dir}")

    # --- 1. Veri Hazırlama: Dosyaları Tara ve Etiketleri Çıkar --- (main bloğu içinde)
    print("Eğitim dosyaları taranıyor ve etiketler çıkarılıyor...")
    all_train_files = glob.glob(os.path.join(TRAIN_DATA_DIR, "*.png"))

    # Etiketleri çıkar ve None olanları filtrele
    valid_files_labels = []
    for f in all_train_files:
        label = parse_image_filename(f)
        if label is not None:
            valid_files_labels.append((f, label))

    if not valid_files_labels:
        raise ValueError(
            f"Eğitim dizininde ('{TRAIN_DATA_DIR}') geçerli etiketli ve okunabilir PNG dosyası bulunamadı.")

    all_train_files = [item[0] for item in valid_files_labels]
    all_train_labels = [item[1] for item in valid_files_labels]

    print(f"Toplam {len(all_train_files)} geçerli eğitim örneği bulundu.")
    initial_class_counts = Counter(all_train_labels)  # Bölmeden önceki dağılım
    print("Başlangıç Sınıf Dağılımı (Tüm Eğitim Verisi):", initial_class_counts)

    # --- 2. Veri Bölme: Stratified Train/Validation Split --- (main bloğu içinde)
    try:
        train_files, val_files, train_labels, val_labels = train_test_split(
            all_train_files,
            all_train_labels,
            test_size=VALIDATION_RATIO,
            stratify=all_train_labels,  # Sınıf oranlarını koru
            random_state=RANDOM_SEED
        )
    except ValueError as e:
        print(
            f"Hata: Stratified split yapılamadı. Muhtemelen bazı sınıfların çok az örneği var: {e}")
        print("Sınıf Dağılımı:", initial_class_counts)
        # Alternatif: Stratify olmadan böl veya az örnekli sınıfları birleştir/çıkar
        raise e  # Şimdilik hatayı yükseltelim

    # Test dosyalarını bul
    test_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.png"))

    print("\n--- Veri Seti Bölme Özeti ---")
    print(f"Oluşturulan Eğitim Seti Boyutu: {len(train_files)}")
    print(f"Oluşturulan Validasyon Seti Boyutu: {len(val_files)}")
    # Sadece bulunan dosya sayısı
    print(f"Bulunan Test Seti Boyutu: {len(test_files)}")
    print("Eğitim Seti Sınıf Dağılımı:", Counter(train_labels))
    print("Validasyon Seti Sınıf Dağılımı:", Counter(val_labels))

    # --- 3. Sınıf->Index Eşlemesi ve Sınıf Ağırlıkları --- (main bloğu içinde)
    # Sınıfları tüm eğitim verisinden alıp sırala (tutarlılık için)
    all_possible_classes = sorted(initial_class_counts.keys())
    num_classes = len(all_possible_classes)
    class_to_idx = {cls_name: i for i,
                    cls_name in enumerate(all_possible_classes)}
    print("Sınıf -> Index Eşlemesi:", class_to_idx)

    # Ağırlıkları EĞİTİM setindeki dağılıma göre hesapla (VİDEODA BAHSET KESİN)
    train_class_counts = Counter(train_labels)
    total_train_samples = len(train_labels)

    weights_list = [0.0] * num_classes
    class_weights_print = {}
    for i, class_name in enumerate(all_possible_classes):
        # Eğitim setindeki sayıyı al (yoksa 0)
        count = train_class_counts.get(class_name, 0)
        if count > 0:
            weight = total_train_samples / (num_classes * count)
            weights_list[i] = weight
            class_weights_print[class_name] = weight
        else:
            # Eğitim setinde olmayan sınıf için ağırlık (örneğin 1.0 veya 0.0)
            # CrossEntropyLoss weight=0 olanı görmezden gelebilir, ama emin olmak lazım.
            # Şimdilik 1.0 atayalım.
            weights_list[i] = 1.0
            class_weights_print[class_name] = 1.0
            print(
                f"Uyarı: '{class_name}' sınıfı eğitim setinde bulunamadı, ağırlık 1.0 olarak ayarlandı.")

    class_weights_tensor = torch.tensor(weights_list, dtype=torch.float)

    print("\n--- Sınıf Ağırlıkları (Eğitim Setine Göre) ---")
    print("Hesaplanan Ağırlıklar:", class_weights_print)
    print("Tensor:", class_weights_tensor)

    # --- 4. Veri Dönüşümleri (Transforms) --- (main bloğu içinde)
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # --- 5. Dataset ve DataLoader Oluşturma --- (main bloğu içinde)
    train_dataset = MedicalImageDataset(
        imagePaths=train_files, class_to_idx=class_to_idx, transform=train_transform)
    val_dataset = MedicalImageDataset(
        imagePaths=val_files, class_to_idx=class_to_idx, transform=val_test_transform)
    test_dataset = MedicalImageDataset(
        imagePaths=test_files, class_to_idx=class_to_idx, transform=val_test_transform)

    # DataLoader'lar
    # Windows'ta num_workers > 0 sorun çıkarabilir, kontrol ekleyelim (deep seek önerdi vardır bir bildiği Robust olacak bu kod)
    num_workers = 2 if os.name == 'posix' else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              # drop_last=True eklenebilir
                              shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    # --- 6. Model, Cihaz, Loss, Optimizer, Scheduler --- (main bloğu içinde)
    model = ModifiedMedicalCNN(
        num_classes=num_classes, dropout_rate=DROPOUT_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCihaz kullanılıyor: {device}")
    model.to(device)

    # Ağırlık tensorunu doğru cihaza gönder
    class_weights_tensor = class_weights_tensor.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # --- TensorBoard Model Grafiği Ekleme ---
    try:
        # DataLoader'dan örnek bir batch al
        dataiter = iter(train_loader)
        # next() kullanırken boş olma durumunu kontrol et
        sample_inputs, sample_labels = next(dataiter, (None, None))
        if sample_inputs is not None:
            # Sadece geçerli input varsa grafiği ekle
            # Dataset'teki hata durumunu kontrol et (opsiyonel)
            if -1 not in sample_labels:
                writer.add_graph(model, sample_inputs.to(device))
                print("Model grafiği TensorBoard'a eklendi.")
            else:
                print(
                    "Uyarı: Örnek batch hatalı etiket içeriyor, model grafiği eklenmedi.")
        else:
            print("Uyarı: Eğitim DataLoader boş, model grafiği eklenemedi.")
    except Exception as e:
        print(f"TensorBoard'a model grafiği eklenemedi: {e}")

    # --- 7. Eğitim Döngüsü ---
    print("\n--- Eğitim Başlıyor ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 20  # Erken durdurma sabrı

    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            # Hatalı yüklenen veriyi kontrol et (label == -1 ise)
            valid_indices = labels != -1
            if not valid_indices.all():
                print(
                    f"Uyarı: Epoch {epoch+1}, batch içinde hatalı yüklenen veri atlanıyor.")
                inputs = inputs[valid_indices]
                labels = labels[valid_indices]
                if inputs.size(0) == 0:  # Eğer tüm batch hatalıysa atla
                    continue

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)  # Kaybı batch boyutuyla çarp

        # Epoch sonu ortalama eğitim kaybı
        epoch_train_loss = running_train_loss / \
            len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Hatalı yüklenen veriyi kontrol et
                valid_indices = labels != -1
                if not valid_indices.all():
                    inputs = inputs[valid_indices]
                    labels = labels[valid_indices]
                    if inputs.size(0) == 0:
                        continue

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Validasyon kaybı için ağırlıksız loss kullanmak yaygındır
                val_loss_batch = nn.CrossEntropyLoss()(outputs, labels)
                running_val_loss += val_loss_batch.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Epoch sonu ortalama validasyon kaybı ve doğruluğu
        epoch_val_loss = running_val_loss / \
            len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0

        # Mevcut öğrenme oranını al
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch+1}/{NUM_EPOCHS} | '
            f'Train Loss: {epoch_train_loss:.4f} | '
            # f'Train Acc: {epoch_train_acc:.4f} | ' # Eğitim doğruluğu eklenirse
            f'Val Loss: {epoch_val_loss:.4f} | '
            f'Val Acc: {epoch_val_acc:.4f} | '
            f'LR: {current_lr:.6f}'
        )

        # TensorBoard'a logla
        writer.add_scalar('Loss/train_epoch', epoch_train_loss, epoch)
        writer.add_scalar('Loss/validation', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', epoch_val_acc, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        # En iyi modeli kaydet (Val Loss'a göre tab2)
        if epoch_val_loss < best_val_loss:
            print(
                f'Validasyon kaybı iyileşti ({best_val_loss:.4f} --> {epoch_val_loss:.4f}). Model kaydediliyor...')
            best_val_loss = epoch_val_loss
            # Model adının test aşamasında kullanılanla aynı olduğundan emin ol
            # Düzeltildi: best_model.pth
            # TO DO Diğerlerini direkt ana yola kaydettim bunu da buraya taşımış olayım runs'ın içinde k
            save_path = os.path.join(log_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f'Model kaydedildi: {save_path}')
            epochs_no_improve = 0  # Sayaç sıfırla
        else:
            epochs_no_improve += 1  # İyileşme yok, sayaç artır

        # Learning rate scheduler'ı güncelle
        scheduler.step(epoch_val_loss)

        # Erken durdurma kontrolü
        if epochs_no_improve >= early_stopping_patience:
            print(
                f'\n{early_stopping_patience} epoch boyunca validasyon kaybında iyileşme olmadı. Eğitim erken durduruluyor.')
            break  # Eğitim döngüsünden çık

    print('\nEğitim tamamlandı!')
    writer.close()  # TensorBoard writer'ı kapat

    # --- 8. Test Aşaması NORMALDE DİĞER DOSYALARDA YAPTIM BU SEFER BURDA HALLEDELİM ---
    print("\nEn iyi model test seti üzerinde değerlendiriliyor...")
    best_model_path = os.path.join(
        log_dir, 'best_model.pth')  # Kaydedilen modelin yolu

    if os.path.exists(best_model_path):
        try:
            # En iyi modeli yükle (aynı model instance üzerine)
            model.load_state_dict(torch.load(best_model_path))
            # Modeli tekrar doğru cihaza gönder (gerekli olabilir)
            model.to(device)
            model.eval()  # Değerlendirme moduna al

            running_test_loss = 0.0
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    # Hatalı yüklenen veriyi kontrol et
                    valid_indices = labels != -1
                    if not valid_indices.all():
                        inputs = inputs[valid_indices]
                        labels = labels[valid_indices]
                        if inputs.size(0) == 0:
                            continue

                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    # Test kaybı (ağırlıksız)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    running_test_loss += loss.item() * inputs.size(0)

                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

                    # Raporlama için tahminleri ve etiketleri topla
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Ortalama test kaybı ve doğruluğu
            avg_test_loss = running_test_loss / \
                len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0
            test_acc = test_correct / test_total if test_total > 0 else 0
            print(
                f'Test Seti Sonuçları -> Loss: {avg_test_loss:.4f} | Accuracy: {test_acc:.4f}')

            # Detaylı Sınıflandırma Raporu (scikit-learn kuruluysa)
            try:
                from sklearn.metrics import classification_report, confusion_matrix
                print("\nSınıflandırma Raporu (Test Seti):")
                # Sınıf isimlerini class_to_idx'den al
                class_names = [k for k, v in sorted(
                    class_to_idx.items(), key=lambda item: item[1])]
                # zero_division=0: Bazı sınıflar hiç tahmin edilmediyse veya hiç yoksa uyarıyı engeller
                print(classification_report(all_labels, all_preds,
                                            target_names=class_names, digits=4, zero_division=0))

                print("\nKarmaşıklık Matrisi (Test Seti):")
                # labels parametresi matrisin sıralamasını garantiler
                cm = confusion_matrix(
                    all_labels, all_preds, labels=list(range(num_classes)))
                print(cm)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names)
                plt.xlabel('Tahmin Edilen Etiket')
                plt.ylabel('Gerçek Etiket')

                # Başlığı doğruluk ve kayıp değerleriyle güncelle
                title = f'Accuracy: {test_acc:.4f}, Loss: {avg_test_loss:.4f}'
                plt.title(title)

                plt.savefig('different_approach__confusion_matrix.png')
                plt.show()
            except ImportError:
                print(
                    "\nDaha detaylı metrikler için 'pip install scikit-learn' yükleyin.")
            except Exception as e:
                print(
                    f"\nSınıflandırma raporu/matrisi oluşturulurken hata: {e}")

        except Exception as e:
            print(f"Test sırasında bir hata oluştu: {e}")
            import traceback
            traceback.print_exc()  # Hatanın detayını yazdır
    else:
        print(
            f"Uyarı: Kaydedilmiş en iyi model bulunamadı ({best_model_path}). Test adımı atlanıyor.")
