import mne
import numpy as np

def load_eeg(fname):
    """EEG faylini yuklash"""
    raw = mne.io.read_raw_edf(fname, preload=True)
    print(f"✅ Yuklandi: {len(raw.ch_names)} kanal")
    return raw

def preprocess_signal(raw):
    """Signallarni tozalash"""
    raw.filter(1, 40)
    return raw