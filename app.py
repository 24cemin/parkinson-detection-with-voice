import streamlit as st
import joblib
import parselmouth
import numpy as np
import os
from utils import extract_features_from_sound
from streamlit_mic_recorder import mic_recorder

# --- Sayfa Ayarları ---
st.set_page_config(page_title="Parkinson AI", page_icon="🧠", layout="centered")

# --- CSS: Buton Tasarımları ---
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Modeli Yükle ---
if os.path.exists("parkinson_model.pkl"):
    model = joblib.load("parkinson_model.pkl")
else:
    st.error("⚠️ Model bulunamadı! Önce eğitimi tamamla.")
    st.stop()

st.title("🧠 Parkinson Erken Teşhis Sistemi")
st.info("Ses analizine dayanarak Parkinson riskini hesaplar.")

# --- SESSION STATE (HAFIZA) AYARLARI ---
# Sayfa yenilense bile veriyi tutmak için burayı kullanıyoruz
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None

# --- SEKMELER ---
tab1, tab2 = st.tabs(["🎙️ Mikrofon", "📂 Dosya Yükle"])

# --- SEKME 1: MİKROFON ---
with tab1:
    st.write("Kaydı başlatın, **'Aaaaa'** deyin ve bitirin.")
    
    # Mikrofon bileşeni (Hafızaya atma işlemi burada)
    audio = mic_recorder(
        start_prompt="⏺️ Kaydı Başlat",
        stop_prompt="⏹️ Kaydı Bitir",
        key='recorder',
        format="wav"
    )
    
    if audio:
        # Sesi diske kaydet
        with open("temp_input.wav", "wb") as f:
            f.write(audio['bytes'])
        
        # Hafızaya dosya yolunu işle
        st.session_state.audio_path = "temp_input.wav"
        st.success("✅ Ses hafızaya alındı! Analiz edebilirsiniz.")
        st.audio(audio['bytes'])

# --- SEKME 2: DOSYA YÜKLEME ---
with tab2:
    uploaded_file = st.file_uploader("WAV Dosyası Seçin", type=["wav"])
    
    if uploaded_file:
        with open("temp_input.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.session_state.audio_path = "temp_input.wav"
        st.success("✅ Dosya yüklendi!")
        st.audio("temp_input.wav")

# --- ANALİZ BÖLÜMÜ ---
st.divider()

# Analiz butonu artık Session State'e bakacak
if st.button("🚀 SONUCU GÖSTER"):
    # Eğer hafızada dosya varsa işlem yap
    if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
        
        with st.spinner('Yapay Zeka sesi inceliyor...'):
            try:
                # 1. Analiz
                sound = parselmouth.Sound(st.session_state.audio_path)
                features = extract_features_from_sound(sound)
                
                if features:
                    # 2. Tahmin
                    # Modelin beklediği format için reshape yapıyoruz
                    features_array = np.array(features).reshape(1, -1)
                    prob = model.predict_proba(features_array)[0]
                    risk_score = prob[1] * 100
                    
                    # 3. Sonuç Ekranı
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Parkinson Riski", f"%{risk_score:.1f}")
                    
                    with col2:
                        if risk_score > 50:
                            st.error("⚠️ SONUÇ: RİSKLİ")
                            st.write("Seste titreme (tremor) bulguları var.")
                        else:
                            st.success("✅ SONUÇ: SAĞLIKLI")
                            st.write("Ses verileri normal.")
                else:
                    st.warning("Ses analiz edilemedi. Lütfen daha net bir kayıt alın.")
                    
            except Exception as e:
                st.error(f"Hata: {e}")
    else:
        st.warning("⚠️ Lütfen önce ses kaydedin veya dosya yükleyin!")