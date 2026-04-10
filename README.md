# NeuroAI EEG Analyzer

NeuroAI EEG Analyzer - shifoxona sharoitida EEG signallarini avtomatik qayta ishlash va shifokor qarorini qo'llab-quvvatlash uchun tayyorlangan MVP platforma.

## Nima qila oladi

- EEG faylini yuklash: `EDF`, `BDF`, `CSV`, `TXT`
- Demo EEG signal yaratish
- Band-pass va notch filtering
- Signal sifati va artefakt ulushini baholash
- EEG ritmlarining nisbiy quvvatini hisoblash
- Shubhali segmentlarni heuristik usulda belgilash
- Risk score va klinik tavsiya chiqarish
- Avtomatik matnli hisobot yaratish

## Klinik maqsad

Platforma quyidagi vazifalar uchun boshlang'ich yechim sifatida mo'ljallangan:

- normal va abnormal EEG ni ajratish
- epileptiform faollikka shubhali segmentlarni topish
- signal sifati past yozuvlarni tez aniqlash
- shifokorga ko'rib chiqish uchun qisqa xulosa tayyorlash

Muhim: bu tizim mustaqil tashxis qo'ymaydi. U faqat `clinical decision support` vositasi sifatida ishlatilishi kerak.

## Arxitektura

1. EEG manbasi: demo yoki yuklangan fayl
2. Preprocessing: de-mean, band-pass, notch
3. Feature extraction: band powers, dominant frequency, artifact ratio
4. Detection: line-length va peak-to-peak asosidagi shubhali segment heuristikasi
5. Clinical output: risk score, tavsiya, report

## Ishga tushirish

1. Kutubxonalarni o'rnating:

```bash
pip install -r requirements.txt
```

2. Streamlit ilovasini ishga tushiring:

```bash
streamlit run app.py
```

3. Demo loginlar:

- `admin / admin123`
- `doctor / doctor123`

## Hozirgi cheklovlar

- AI qismi hozircha to'liq o'qitilgan model emas, balki klinik MVP uchun heuristik tahlil
- PDF report yo'q, hisobot `.txt` ko'rinishida yuklanadi
- HIS/EMR integratsiyasi hali qo'shilmagan
- Annotatsiyalangan klinik dataset bilan validatsiya hali bajarilmagan

## Keyingi bosqichlar

- `MNE` asosida rivojlangan artefakt removal
- o'qitilgan `1D CNN` yoki `CNN-LSTM` modelini ulash
- shifokor tasdiqlash oqimi va audit log
- bemor tarixi va ma'lumotlar bazasi
- PDF hisobot va rolga asoslangan boshqaruv
