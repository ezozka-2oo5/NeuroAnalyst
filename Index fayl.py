# icons.py
import os

print("Icons papkasi yaratilmoqda...")

# icons papkasini yaratish
os.makedirs('icons', exist_ok=True)

# Oddiy logotip fayli yaratish
with open('icons/logo.txt', 'w') as f:
    f.write('NeuroAnalyst - EEG Tahlil Platformasi')

print("✅ icons papkasi va logo.txt yaratildi!")
