import parselmouth
from parselmouth.praat import call
import numpy as np

def extract_features_from_sound(sound):
    try:
        pitch = sound.to_pitch()
        pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
        
        # Jitter
        jitter_local = call(pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_local_abs = call(pulses, "Get jitter (local, absolute)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_rap = call(pulses, "Get jitter (rap)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter_ppq5 = call(pulses, "Get jitter (ppq5)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        
        # Shimmer
        shimmer_local = call([sound, pulses], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_local_db = call([sound, pulses], "Get shimmer (local_dB)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = call([sound, pulses], "Get shimmer (apq3)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = call([sound, pulses], "Get shimmer (apq5)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        
        # HNR
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        features = [
            jitter_local * 100, jitter_local_abs, jitter_rap * 100, jitter_ppq5 * 100,
            shimmer_local * 100, shimmer_local_db, shimmer_apq3 * 100, shimmer_apq5 * 100, hnr
        ]
        return [0 if np.isnan(x) else x for x in features]
    except:
        return None