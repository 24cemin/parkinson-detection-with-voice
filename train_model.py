import os
import glob
import parselmouth
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils import extract_features_from_sound

# Klasör Ayarları
BASE_PATH = "dataset"
HEALTHY_PATH = os.path.join(BASE_PATH, "healthy")
PARKINSON_PATH = os.path.join(BASE_PATH, "parkinson")

data = []
labels = []

def process_folder(folder_path, label):
    files = glob.glob(os.path.join(folder_path, "*.wav"))
    local_count = 0
    print(f" İşleniyor: {folder_path} ({len(files)} dosya)")
    
    for file in files:
        try:
            sound = parselmouth.Sound(file)
            dur = sound.get_total_duration()
            
            if dur < 1.0: continue # Çok kısaları atla

            # --- STRATEJİ: 3 PARÇAYA BÖLME (DATA AUGMENTATION) ---
            # 1000 veriyi geçmek için her sesi Baş, Orta, Son olarak alıyoruz.
            t1 = dur / 3
            t2 = (dur / 3) * 2
            
            parts = [
                sound.extract_part(0, t1),      # 1. Parça
                sound.extract_part(t1, t2),     # 2. Parça
                sound.extract_part(t2, dur)     # 3. Parça
            ]
            
            for part in parts:
                feats = extract_features_from_sound(part)
                if feats:
                    data.append(feats)
                    labels.append(label)
                    local_count += 1
                    
        except Exception as e:
            print(f"Hata: {file} - {e}")
            
    return local_count

# Verileri İşle
print("Veri seti hazırlanıyor, lütfen bekleyin...")
h_num = process_folder(HEALTHY_PATH, 0)
p_num = process_folder(PARKINSON_PATH, 1)

total = h_num + p_num
print(f"\n İŞLEM TAMAM!")
print(f" Sağlıklı Parça Sayısı: {h_num}")
print(f" Parkinson Parça Sayısı: {p_num}")
print(f" TOPLAM VERİ SETİ BOYUTU: {total}")

if total < 100:
    print("HATA: Yeterli veri bulunamadı! Klasör isimlerini kontrol et.")
    exit()

# Eğitim
print("\n Model eğitiliyor...")
cols = ["Jitter_perc", "Jitter_Abs", "Jitter_RAP", "Jitter_PPQ5", 
        "Shimmer_perc", "Shimmer_dB", "Shimmer_APQ3", "Shimmer_APQ5", "HNR"]
X = pd.DataFrame(data, columns=cols)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42) # Daha güçlü bir orman
model.fit(X_train, y_train)

acc = accuracy_score(y_test, y_pred := model.predict(X_test))
print(f"\n MODEL BAŞARISI: %{acc * 100:.2f}")
print(classification_report(y_test, y_pred))

joblib.dump(model, "parkinson_model.pkl")
print(" Model kaydedildi: parkinson_model.pkl")