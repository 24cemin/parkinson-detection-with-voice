#  Parkinson's Disease Detection System via Voice Analysis

An end-to-end Machine Learning pipeline designed to detect early signs of Parkinson's Disease using vocal acoustic analysis. This project leverages Signal Processing (Praat/Parselmouth) and a Random Forest Classifier to distinguish between healthy and Parkinsonian subjects based on raw audio recordings.

##  Project Overview

This system analyzes sustained phonation (vowel "Aaaaa") recordings to extract micro-tremors and vocal instabilities. It features a custom **Data Augmentation** strategy (Time-Domain Slicing) and a user-friendly Web UI for real-time testing.

### Key Features
* **Signal Processing:** Extraction of Jitter, Shimmer, and HNR features from raw `.wav` files.
* **Data Engineering:** Implemented a slicing technique to triple the dataset size (from ~500 to 1485+ samples) for robust training.
* **Machine Learning:** Random Forest Classifier achieving **~82% Accuracy**.
* **Interactive UI:** Built with **Streamlit**, supporting both file upload and real-time microphone recording.


##  Dataset & Citation

The model is trained on the **Italian Parkinson's Voice and Speech Dataset**. We strictly utilized the sustained vowel recordings from elderly healthy controls and Parkinson's patients to avoid age bias.

** Dataset Download:**
Due to repository size limits and professional practices, the raw audio files are not included in this repo. Please download them from the official IEEE Dataport:

> **[Download Dataset Here (IEEE Dataport)](https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech)**

** Citation:**
If you use this project or the data, please cite the original authors:
> Giovanni Dimauro, Francesco Girardi, "Italian Parkinson's Voice and Speech", IEEE Dataport, June 11, 2019, doi:10.21227/aw6b-tg17.


##  Installation & Setup (IMPORTANT)

### 1. Clone the repository
    git clone https://github.com/24cemin/parkinson-detection.git
    cd parkinson-detection

### 2. Install dependencies
    pip install -r requirements.txt

### 3. Dataset Preparation (Critical Step) 
Since the raw dataset contains various speech tasks and age groups, you must perform the following cleaning steps to replicate our results:

1.  **Download & Extract:** Download the dataset from the link above.
2.  **Filter Subjects (Age Bias Removal):**
    * **DELETE** the folder containing "Young Healthy Controls". (We only compare Elderly Healthy vs. Parkinson's Patients to ensure age-matched analysis).
3.  **Filter Audio Tasks (Signal Stability):**
    * Inside the remaining folders, **DELETE** any files related to:
        * Sentence Reading (usually labeled as `B1`, `B2`, `FB`...)
        * Words/Syllables (labeled as `D1`, `D2`...)
    * **KEEP ONLY** the sustained vowels (usually labeled as `VA`, `VE`, `VI`, `VO`, `VU` -> e.g., `VA1xxx.wav`).
4.  **Final Folder Structure:**
    Create a `dataset` folder in the project root and organize the cleaned files as follows:

    parkinson-detection/
    │
    ├── dataset/
    │   ├── healthy/     <-- (Put cleaned Elderly Healthy .wav files here)
    │   └── parkinson/   <-- (Put cleaned Parkinson .wav files here)
    │
    ├── train_model.py
    ├── app.py
    └── ...

---

##  Usage

### 1. Train the Model
Once the dataset is organized as shown above, run the training script. This will process the audio, apply data augmentation (slicing), and save the model as `parkinson_model.pkl`.

    python train_model.py

*Expected Output:* The script should process approx. 1400+ audio slices and achieve ~80-85% accuracy.

### 2. Run the Web App
Launch the interface to test the model (using the pre-trained `.pkl` file or your newly trained one).

    streamlit run app.py


## Disclaimer
This project is for **educational and research purposes only**. It is not a certified medical device and should not be used for definitive diagnosis.
