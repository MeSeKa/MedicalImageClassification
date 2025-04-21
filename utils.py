import os
import re


def parse_image_filename(filename):
    """
    Dosya adından sınıf etiketini ('0', '1+', '2+', '3+') çıkarır.
    Örnek: '00004_train_1+.png' -> '1+'
           '000044_test_2+.png' -> '2+'
           '000044_train_0.png' -> '0'
    """
    # Dosya adının sonundaki sınıf etiketini ve .png uzantısını arar
    # _ ile başlayan, ardından 0-3 arası bir rakam ve opsiyonel '+' içeren kısmı arar
    match = re.search(r"_([0-3]\+?)\.png$", os.path.basename(filename))
    if match:
        return match.group(1)  # Yakalanan grup ('0', '1+', '2+', '3+')
    else:
        # Eğer format eşleşmezse uyarı verip None dönebiliriz
        print(
            f"Uyarı: Dosya adı formatı anlaşılamadı veya sınıf etiketi bulunamadı: {filename}"
        )
        return None
