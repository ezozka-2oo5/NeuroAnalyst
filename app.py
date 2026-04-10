import hashlib
import io
import os
import sqlite3
import tempfile
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MPL_CONFIG_DIR = os.path.join(BASE_DIR, ".mplconfig")
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy.signal import butter, filtfilt, iirnotch, spectrogram, welch

try:
    import mne
except ImportError:  # pragma: no cover - optional dependency
    mne = None


DB_PATH = os.path.join(BASE_DIR, "clinical_platform.db")
MODEL_VERSION = "NeuroAI Clinical Heuristic v2.1"
BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Beta": (12.0, 30.0),
    "Gamma": (30.0, 45.0),
}

DEMO_USERS = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode("utf-8")).hexdigest(),
        "role": "admin",
        "name": "Administrator",
    },
    "doctor": {
        "password_hash": hashlib.sha256("doctor123".encode("utf-8")).hexdigest(),
        "role": "doctor",
        "name": "Shifokor",
    },
}

DEMO_CHANNELS = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "F3-C3",
    "C3-P3",
    "F4-C4",
    "C4-P4",
]

st.set_page_config(
    page_title="NeuroAI EEG Analyzer",
    page_icon="N",
    layout="wide",
)


def init_session_state():
    defaults = {
        "authenticated": False,
        "user": None,
        "patient_name": "Test Bemor",
        "patient_age": 34,
        "patient_gender": "Erkak",
        "patient_id": "EEG-001",
        "department": "Nevrologiya",
        "clinical_note": "Tutqanoq shubhasi / EEG nazorati.",
        "technician": "EEG Operator",
        "study_priority": "Routine",
        "raw_recording": None,
        "clean_recording": None,
        "analysis": None,
        "report_text": "",
        "report_pdf": None,
        "saved_patient_db_id": None,
        "last_saved_study_id": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_connection():
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_code TEXT UNIQUE,
            full_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            department TEXT,
            clinical_note TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS studies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            source_name TEXT,
            source_type TEXT,
            duration_sec REAL,
            sfreq REAL,
            channel_count INTEGER,
            signal_quality REAL,
            artifact_ratio REAL,
            seizure_risk REAL,
            label TEXT,
            confidence REAL,
            dominant_channel TEXT,
            recommendation TEXT,
            report_text TEXT,
            created_at TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            action TEXT,
            detail TEXT,
            created_at TEXT
        )
        """
    )
    connection.commit()
    connection.close()


def log_audit(action, detail):
    user = st.session_state.get("user")
    username = user["username"] if user else "anonymous"
    connection = get_connection()
    connection.execute(
        "INSERT INTO audit_logs (username, action, detail, created_at) VALUES (?, ?, ?, ?)",
        (username, action, detail, datetime.now().isoformat(timespec="seconds")),
    )
    connection.commit()
    connection.close()


def save_patient(patient):
    now = datetime.now().isoformat(timespec="seconds")
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO patients (patient_code, full_name, age, gender, department, clinical_note, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(patient_code) DO UPDATE SET
            full_name = excluded.full_name,
            age = excluded.age,
            gender = excluded.gender,
            department = excluded.department,
            clinical_note = excluded.clinical_note,
            updated_at = excluded.updated_at
        """,
        (
            patient["patient_id"],
            patient["name"],
            patient["age"],
            patient["gender"],
            patient["department"],
            patient["clinical_note"],
            now,
            now,
        ),
    )
    connection.commit()
    row = connection.execute(
        "SELECT id FROM patients WHERE patient_code = ?",
        (patient["patient_id"],),
    ).fetchone()
    connection.close()
    log_audit("save_patient", patient["patient_id"])
    return int(row["id"])


def save_study(patient_db_id, recording, analysis, report_text):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO studies (
            patient_id, source_name, source_type, duration_sec, sfreq, channel_count,
            signal_quality, artifact_ratio, seizure_risk, label, confidence,
            dominant_channel, recommendation, report_text, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_db_id,
            recording["source_name"],
            recording["source_type"],
            recording["data"].shape[1] / recording["sfreq"],
            recording["sfreq"],
            recording["data"].shape[0],
            analysis["signal_quality"],
            analysis["artifact_ratio"],
            analysis["risk_score"],
            analysis["label"],
            analysis["confidence"],
            analysis["dominant_channel"],
            analysis["recommendation"],
            report_text,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    connection.commit()
    study_id = cursor.lastrowid
    connection.close()
    log_audit("save_study", f"study_id={study_id}")
    return study_id


def load_patients():
    connection = get_connection()
    dataframe = pd.read_sql_query(
        """
        SELECT id, patient_code, full_name, age, gender, department, clinical_note, updated_at
        FROM patients
        ORDER BY updated_at DESC
        """,
        connection,
    )
    connection.close()
    return dataframe


def load_recent_studies(limit=25):
    connection = get_connection()
    dataframe = pd.read_sql_query(
        f"""
        SELECT s.id, p.patient_code, p.full_name, s.label, s.confidence, s.signal_quality,
               s.artifact_ratio, s.seizure_risk, s.dominant_channel, s.created_at
        FROM studies s
        JOIN patients p ON p.id = s.patient_id
        ORDER BY s.created_at DESC
        LIMIT {int(limit)}
        """,
        connection,
    )
    connection.close()
    return dataframe


def load_audit_logs(limit=40):
    connection = get_connection()
    dataframe = pd.read_sql_query(
        f"SELECT username, action, detail, created_at FROM audit_logs ORDER BY created_at DESC LIMIT {int(limit)}",
        connection,
    )
    connection.close()
    return dataframe


def inject_styles():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
            }
            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, #0f1c2f 0%, #14263f 100%);
                border-right: 1px solid rgba(125, 211, 252, 0.18);
            }
            .hero-shell {
                background:
                    radial-gradient(circle at 15% 10%, rgba(56, 189, 248, 0.18), transparent 28%),
                    radial-gradient(circle at 85% 15%, rgba(16, 185, 129, 0.14), transparent 24%),
                    linear-gradient(140deg, #0b1220 0%, #132238 56%, #10283a 100%);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 26px;
                padding: 28px 32px;
                margin-bottom: 1rem;
                box-shadow: 0 20px 60px rgba(2, 6, 23, 0.28);
            }
            .hero-title {
                color: #f8fafc;
                font-size: 2.45rem;
                font-weight: 800;
                letter-spacing: -0.02em;
                margin-bottom: 0.2rem;
            }
            .hero-subtitle {
                color: #cbd5e1;
                font-size: 1rem;
                max-width: 70rem;
            }
            .glass-card {
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(17, 24, 39, 0.94));
                border: 1px solid rgba(56, 189, 248, 0.18);
                border-radius: 18px;
                padding: 14px 16px;
                margin-bottom: 0.8rem;
            }
            .glass-label {
                color: #94a3b8;
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .glass-value {
                color: #f8fafc;
                font-size: 1.28rem;
                font-weight: 700;
            }
            .section-kicker {
                color: #38bdf8;
                font-size: 0.86rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.2rem;
            }
            .section-title {
                color: #e2e8f0;
                font-size: 1.45rem;
                font-weight: 700;
                margin-bottom: 0.6rem;
            }
            .clinical-note {
                background: linear-gradient(90deg, rgba(6, 78, 59, 0.75), rgba(15, 118, 110, 0.55));
                border: 1px solid rgba(45, 212, 191, 0.2);
                border-radius: 18px;
                padding: 14px 16px;
                color: #ecfeff;
                margin: 0.7rem 0 1rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value):
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="glass-label">{label}</div>
            <div class="glass-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_password():
    if st.session_state.get("authenticated"):
        return True

    st.sidebar.header("Kirish")
    username = st.sidebar.text_input("Foydalanuvchi nomi")
    password = st.sidebar.text_input("Parol", type="password")

    if st.sidebar.button("Kirish", use_container_width=True):
        user = DEMO_USERS.get(username)
        password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        if user and password_hash == user["password_hash"]:
            st.session_state.authenticated = True
            st.session_state.user = {
                "username": username,
                "role": user["role"],
                "name": user["name"],
            }
            log_audit("login", f"user={username}")
            st.rerun()
        st.sidebar.error("Login yoki parol xato.")

    st.sidebar.caption("Demo hisoblar: admin / admin123, doctor / doctor123")
    return False


def normalize_orientation(data):
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        return array[np.newaxis, :]
    if array.shape[0] > 64 and array.shape[1] <= 64:
        return array.T
    if array.shape[0] > array.shape[1] and array.shape[1] <= 32:
        return array.T
    return array


def create_demo_signal(duration_sec=20, sfreq=256, n_channels=12, abnormal=False, seed=11):
    rng = np.random.default_rng(seed)
    times = np.arange(0, duration_sec, 1 / sfreq)
    data = []
    channel_names = DEMO_CHANNELS[:n_channels]

    for index in range(n_channels):
        delta = 0.55 * np.sin(2 * np.pi * 2.0 * times + index / 6)
        theta = (0.25 + index * 0.015) * np.sin(2 * np.pi * 6.0 * times + index / 5)
        alpha = (1.2 - index * 0.03) * np.sin(2 * np.pi * 10.0 * times + index / 8)
        beta = (0.3 + index * 0.02) * np.sin(2 * np.pi * 19.0 * times + index / 4)
        drift = 0.18 * np.sin(2 * np.pi * 0.25 * times + index)
        noise = 0.18 * rng.standard_normal(times.size)
        channel = delta + theta + alpha + beta + drift + noise

        if abnormal and index in (1, 5, 9):
            for event_center in (4.8, 10.9, 15.5):
                spike = np.exp(-((times - event_center) / 0.032) ** 2) * (4.8 + index * 0.15)
                slow_wave = np.exp(-((times - event_center - 0.12) / 0.18) ** 2) * 1.8
                channel += spike + slow_wave

        if index == 0:
            blink = np.exp(-((times - 7.0) / 0.20) ** 2) * 3.2
            channel += blink

        data.append(channel * 32.0)

    return {
        "data": np.asarray(data),
        "sfreq": float(sfreq),
        "channel_names": channel_names,
        "source_name": "Demo EEG",
        "source_type": "demo",
    }


def load_csv_recording(uploaded_file, sfreq):
    file_bytes = uploaded_file.getvalue()
    text = file_bytes.decode("utf-8", errors="ignore")
    csv_buffer = io.StringIO(text.replace(";", ","))
    dataframe = pd.read_csv(csv_buffer, header=None)
    data = normalize_orientation(dataframe.to_numpy(dtype=float))
    return {
        "data": data,
        "sfreq": float(sfreq),
        "channel_names": [f"Ch-{index + 1}" for index in range(data.shape[0])],
        "source_name": uploaded_file.name,
        "source_type": "csv",
    }


def load_edf_recording(uploaded_file):
    if mne is None:
        raise RuntimeError("EDF/BDF yuklash uchun MNE kutubxonasi kerak. Hozir bu muhitda u topilmadi.")

    suffix = "." + uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:
        raw = mne.io.read_raw_edf(temp_path, preload=True, verbose="ERROR")
        data = raw.get_data() * 1e6
        return {
            "data": normalize_orientation(data),
            "sfreq": float(raw.info["sfreq"]),
            "channel_names": list(raw.ch_names),
            "source_name": uploaded_file.name,
            "source_type": "edf",
        }
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def load_recording(uploaded_file, csv_sfreq):
    name = uploaded_file.name.lower()
    if name.endswith((".edf", ".bdf")):
        return load_edf_recording(uploaded_file)
    if name.endswith((".csv", ".txt")):
        return load_csv_recording(uploaded_file, csv_sfreq)
    raise ValueError("Faqat EDF, BDF, CSV yoki TXT fayllar qabul qilinadi.")


def bandpass_filter(data, sfreq, low_cut, high_cut):
    nyquist = sfreq / 2
    low = max(0.1, low_cut)
    high = min(high_cut, nyquist - 0.5)
    if low >= high:
        return data.copy()
    b, a = butter(4, [low, high], btype="band", fs=sfreq)
    return filtfilt(b, a, data, axis=1)


def notch_filter(data, sfreq, notch_hz):
    if notch_hz <= 0 or notch_hz >= sfreq / 2:
        return data.copy()
    b, a = iirnotch(notch_hz, 30, fs=sfreq)
    return filtfilt(b, a, data, axis=1)


def preprocess_recording(recording, low_cut, high_cut, notch_hz, apply_notch):
    data = recording["data"]
    demeaned = data - np.mean(data, axis=1, keepdims=True)
    filtered = bandpass_filter(demeaned, recording["sfreq"], low_cut, high_cut)
    if apply_notch:
        filtered = notch_filter(filtered, recording["sfreq"], notch_hz)
    return {**recording, "data": filtered}


def compute_band_table(data, sfreq, channel_names):
    frequencies, psd = welch(data, fs=sfreq, nperseg=min(1024, data.shape[1]), axis=1)
    rows = []
    for index, channel_name in enumerate(channel_names):
        channel_total = 0.0
        channel_values = {}
        for band_name, (low, high) in BANDS.items():
            mask = (frequencies >= low) & (frequencies < high)
            band_power = float(np.trapezoid(psd[index, mask], frequencies[mask])) if np.any(mask) else 0.0
            channel_values[band_name] = band_power
            channel_total += band_power
        channel_total = channel_total or 1.0
        row = {"Channel": channel_name}
        for band_name in BANDS:
            row[band_name] = channel_values[band_name] / channel_total * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def compute_quality_metrics(data, sfreq, notch_hz):
    centered = data - np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(centered), axis=1, keepdims=True) + 1e-6
    robust_z = 0.6745 * centered / mad
    artifact_ratio = float(np.mean(np.abs(robust_z) > 6.0))

    frequencies, psd = welch(data, fs=sfreq, nperseg=min(1024, data.shape[1]), axis=1)
    avg_psd = np.mean(psd, axis=0)

    def band_integral(low, high):
        mask = (frequencies >= low) & (frequencies < high)
        return float(np.trapezoid(avg_psd[mask], frequencies[mask])) if np.any(mask) else 0.0

    total_power = band_integral(0.5, 45.0) or 1.0
    line_noise = band_integral(max(notch_hz - 1.5, 0.5), min(notch_hz + 1.5, sfreq / 2 - 0.1)) / total_power
    drift = band_integral(0.1, 0.5) / total_power
    emg_noise = band_integral(25.0, min(45.0, sfreq / 2 - 0.1)) / total_power
    flat_channels = int(np.sum(np.std(data, axis=1) < 3.0))

    signal_quality = float(
        np.clip(
            100.0
            - artifact_ratio * 100.0
            - line_noise * 70.0
            - drift * 60.0
            - emg_noise * 35.0
            - flat_channels * 3.0,
            5.0,
            99.0,
        )
    )

    return {
        "artifact_ratio": artifact_ratio * 100.0,
        "line_noise_ratio": line_noise * 100.0,
        "drift_ratio": drift * 100.0,
        "emg_ratio": emg_noise * 100.0,
        "flat_channels": flat_channels,
        "signal_quality": signal_quality,
    }


def detect_events(data, sfreq):
    window_sec = 2.0
    step_sec = 0.5
    window_size = max(int(window_sec * sfreq), 16)
    step_size = max(int(step_sec * sfreq), 1)
    events = []

    for start in range(0, data.shape[1] - window_size + 1, step_size):
        end = start + window_size
        chunk = data[:, start:end]
        peak_to_peak = np.ptp(chunk, axis=1)
        line_length = np.mean(np.abs(np.diff(chunk, axis=1)), axis=1)
        high_slope = np.max(np.abs(np.diff(chunk, axis=1)), axis=1)
        event_score = peak_to_peak * 0.22 + line_length * 1.8 + high_slope * 0.07
        channel_index = int(np.argmax(event_score))
        events.append(
            {
                "start": start / sfreq,
                "end": end / sfreq,
                "score": float(np.max(event_score)),
                "channel_index": channel_index,
                "peak_to_peak": float(peak_to_peak[channel_index]),
                "line_length": float(line_length[channel_index]),
            }
        )

    if not events:
        return []

    scores = np.asarray([event["score"] for event in events], dtype=float)
    threshold = float(np.percentile(scores, 93))
    return [event for event in events if event["score"] >= threshold]


def compute_dominant_frequency(data, sfreq):
    frequencies, psd = welch(data, fs=sfreq, nperseg=min(1024, data.shape[1]), axis=1)
    avg_psd = np.mean(psd, axis=0)
    mask = (frequencies >= 1.0) & (frequencies <= 30.0)
    return float(frequencies[mask][np.argmax(avg_psd[mask])])


def build_analysis(recording, notch_hz):
    data = recording["data"]
    sfreq = recording["sfreq"]
    channel_names = recording["channel_names"]
    band_table = compute_band_table(data, sfreq, channel_names)
    band_means = {band: float(band_table[band].mean()) for band in BANDS}
    quality = compute_quality_metrics(data, sfreq, notch_hz)
    events = detect_events(data, sfreq)
    dominant_frequency = compute_dominant_frequency(data, sfreq)

    slowing_index = (band_means["Delta"] + band_means["Theta"]) / max(band_means["Alpha"] + band_means["Beta"], 1e-6)
    epileptiform_index = float(np.mean([event["score"] for event in events])) if events else 0.0
    dominant_channel = "Aniqlanmadi"
    if events:
        counts = {}
        for event in events:
            counts[event["channel_index"]] = counts.get(event["channel_index"], 0) + 1
        dominant_channel = channel_names[max(counts, key=counts.get)]

    if quality["signal_quality"] < 55:
        label = "Signal sifati past"
        recommendation = "Elektrod kontaktlarini va yozuv sifatini tekshirib, qayta EEG tavsiya etiladi."
        confidence = 48.0
    elif len(events) >= 4 and epileptiform_index > 26:
        label = "Epileptiform faollikka kuchli shubha"
        recommendation = "Shubhali segmentlarni nevrolog tomonidan tezkor vizual verifikatsiya qilish kerak."
        confidence = min(96.0, 72.0 + len(events) * 3.0)
    elif slowing_index > 1.85 and band_means["Alpha"] < 18.0:
        label = "Diffuz sekinlashuv ehtimoli"
        recommendation = "Ensefalopatiya yoki diffuz disfunktsiya uchun klinik baholash tavsiya etiladi."
        confidence = min(90.0, 63.0 + slowing_index * 8.0)
    elif 8.0 <= dominant_frequency <= 12.5 and quality["artifact_ratio"] < 12.0:
        label = "Ehtimol normal EEG"
        recommendation = "Vizual EEG ko'rigi bilan tasdiqlashdan so'ng yakuniy xulosa beriladi."
        confidence = max(76.0, 94.0 - quality["artifact_ratio"] * 0.9)
    else:
        label = "Chegaraviy topilmalar"
        recommendation = "Qo'shimcha klinik kontekst va qo'lda EEG ko'rigi tavsiya etiladi."
        confidence = 67.0

    risk_score = float(
        np.clip(
            confidence * 0.52
            + len(events) * 7.5
            + slowing_index * 10.0
            + max(0.0, 70.0 - quality["signal_quality"]) * 0.35,
            0.0,
            100.0,
        )
    )
    seizure_probability = float(np.clip(len(events) * 11.0 + epileptiform_index * 0.55, 0.0, 100.0))

    return {
        "label": label,
        "confidence": float(confidence),
        "recommendation": recommendation,
        "signal_quality": quality["signal_quality"],
        "artifact_ratio": quality["artifact_ratio"],
        "line_noise_ratio": quality["line_noise_ratio"],
        "drift_ratio": quality["drift_ratio"],
        "emg_ratio": quality["emg_ratio"],
        "flat_channels": quality["flat_channels"],
        "band_table": band_table,
        "band_means": band_means,
        "events": events,
        "dominant_frequency": dominant_frequency,
        "dominant_channel": dominant_channel,
        "slowing_index": float(slowing_index),
        "epileptiform_index": epileptiform_index,
        "risk_score": risk_score,
        "seizure_probability": seizure_probability,
    }


def plot_signal_review(raw_recording, clean_recording, events, window_start_sec=0.0, preview_seconds=12, channel_limit=8):
    sfreq = clean_recording["sfreq"]
    total_samples = clean_recording["data"].shape[1]
    start_sample = max(0, min(int(window_start_sec * sfreq), total_samples - 1))
    preview_samples = min(int(preview_seconds * sfreq), total_samples - start_sample)
    end_sample = start_sample + preview_samples
    times = np.arange(start_sample, end_sample) / sfreq
    n_channels = min(channel_limit, clean_recording["data"].shape[0])

    figure, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for axis, recording, title, color in (
        (axes[0], raw_recording, "Xom signal", "#38bdf8"),
        (axes[1], clean_recording, "Filtrlangan signal", "#34d399"),
    ):
        preview = recording["data"][:n_channels, start_sample:end_sample]
        centered = preview - np.median(preview, axis=1, keepdims=True)
        scales = np.percentile(np.abs(centered), 95, axis=1) + 1e-6
        for index in range(n_channels):
            offset = (n_channels - index) * 4.0
            axis.plot(times, centered[index] / scales[index] + offset, linewidth=0.9, color=color, alpha=0.85)
            axis.text(times[-1] + 0.1, offset, recording["channel_names"][index], fontsize=8, color="#cbd5e1")

        for event in events:
            overlap_start = max(event["start"], window_start_sec)
            overlap_end = min(event["end"], window_start_sec + preview_seconds)
            if overlap_start < overlap_end:
                axis.axvspan(overlap_start, overlap_end, color="#f97316", alpha=0.13)

        axis.set_title(title)
        axis.set_ylabel("Norm + offset")
        axis.grid(alpha=0.18)

    axes[1].set_xlabel("Vaqt (sekund)")
    figure.tight_layout()
    return figure


def plot_band_heatmap(band_table):
    matrix = band_table[[band for band in BANDS]].to_numpy()
    figure, axis = plt.subplots(figsize=(10, 5))
    heatmap = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_yticks(range(len(band_table)))
    axis.set_yticklabels(band_table["Channel"])
    axis.set_xticks(range(len(BANDS)))
    axis.set_xticklabels(list(BANDS.keys()))
    axis.set_title("Kanal bo'yicha ritm ulushlari (%)")
    figure.colorbar(heatmap, ax=axis, shrink=0.88)
    figure.tight_layout()
    return figure


def plot_event_timeline(events, total_duration=None, window_start_sec=0.0, window_duration_sec=12.0):
    figure, axis = plt.subplots(figsize=(10, 3.6))
    if events:
        centers = [(event["start"] + event["end"]) / 2 for event in events]
        scores = [event["score"] for event in events]
        axis.bar(centers, scores, width=0.35, color="#f97316", alpha=0.9)
        axis.set_ylabel("Score")
    else:
        axis.text(0.5, 0.5, "Shubhali segment aniqlanmadi", ha="center", va="center", transform=axis.transAxes)
    if total_duration is not None:
        axis.set_xlim(0, total_duration)
    axis.axvspan(window_start_sec, window_start_sec + window_duration_sec, color="#38bdf8", alpha=0.14)
    axis.set_xlabel("Vaqt (sekund)")
    axis.set_title("Patologik segment timeline")
    axis.grid(alpha=0.18)
    figure.tight_layout()
    return figure


def plot_spectrogram(clean_recording, channel_index=0, window_start_sec=0.0, window_duration_sec=30.0):
    sfreq = clean_recording["sfreq"]
    total_samples = clean_recording["data"].shape[1]
    start_sample = max(0, min(int(window_start_sec * sfreq), total_samples - 1))
    window_samples = min(int(window_duration_sec * sfreq), total_samples - start_sample)
    end_sample = start_sample + window_samples
    data = clean_recording["data"][channel_index, start_sample:end_sample]
    sfreq = clean_recording["sfreq"]
    frequencies, times, spectrum = spectrogram(data, fs=sfreq, nperseg=min(256, len(data)), noverlap=64)
    mask = frequencies <= 45.0

    figure, axis = plt.subplots(figsize=(10, 4))
    image = axis.pcolormesh(times, frequencies[mask], 10 * np.log10(spectrum[mask] + 1e-9), shading="gouraud", cmap="magma")
    axis.set_title(f"Spektrogramma: {clean_recording['channel_names'][channel_index]} ({window_start_sec:.1f}-{window_start_sec + window_duration_sec:.1f}s)")
    axis.set_ylabel("Hz")
    axis.set_xlabel("Vaqt (sekund)")
    figure.colorbar(image, ax=axis, label="dB")
    figure.tight_layout()
    return figure


def build_report_text(patient, recording, analysis):
    event_lines = []
    for index, event in enumerate(analysis["events"][:6], start=1):
        channel_name = recording["channel_names"][event["channel_index"]]
        event_lines.append(
            f"{index}. {event['start']:.1f}-{event['end']:.1f} s | {channel_name} | score {event['score']:.2f}"
        )
    if not event_lines:
        event_lines.append("Shubhali segment aniqlanmadi.")

    lines = [
        "NEUROAI EEG ANALYZER - CLINICAL DECISION SUPPORT REPORT",
        "=" * 64,
        f"Hisobot vaqti: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "BEMOR",
        f"Ism: {patient['name']}",
        f"Yosh: {patient['age']}",
        f"Jins: {patient['gender']}",
        f"ID: {patient['patient_id']}",
        f"Bo'lim: {patient['department']}",
        f"Texnik: {patient['technician']}",
        f"Prioritet: {patient['study_priority']}",
        f"Klinik izoh: {patient['clinical_note']}",
        "",
        "YOZUV",
        f"Manba: {recording['source_name']}",
        f"Format: {recording['source_type']}",
        f"Kanallar: {recording['data'].shape[0]}",
        f"Sampling rate: {recording['sfreq']:.1f} Hz",
        f"Davomiylik: {recording['data'].shape[1] / recording['sfreq']:.1f} s",
        "",
        "AI XULOSA",
        f"Holat: {analysis['label']}",
        f"Ishonch: {analysis['confidence']:.1f}%",
        f"Signal sifati: {analysis['signal_quality']:.1f}%",
        f"Artefakt ulushi: {analysis['artifact_ratio']:.1f}%",
        f"Seizure risk score: {analysis['risk_score']:.1f}/100",
        f"Seizure probability: {analysis['seizure_probability']:.1f}%",
        f"Dominant chastota: {analysis['dominant_frequency']:.2f} Hz",
        f"Dominant kanal: {analysis['dominant_channel']}",
        f"Slowing index: {analysis['slowing_index']:.2f}",
        "",
        "RITM ULUSHLARI (%)",
    ]

    for band_name in BANDS:
        lines.append(f"{band_name}: {analysis['band_means'][band_name]:.1f}")

    lines.extend(["", "SHUBHALI SEGMENTLAR", *event_lines, "", "TAVSIYA", analysis["recommendation"], ""])
    lines.extend(
        [
            "MUHIM ESLATMA",
            "Tizim klinik qarorni qo'llab-quvvatlash uchun mo'ljallangan.",
            "Yakuniy tashxis mutaxassis shifokor tomonidan vizual ko'rik va klinik kontekst asosida beriladi.",
        ]
    )
    return "\n".join(lines)


def build_report_pdf(patient, recording, analysis, report_text):
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title_style",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=8,
    )
    section_style = ParagraphStyle(
        "section_style",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=colors.HexColor("#0f766e"),
        spaceAfter=5,
        spaceBefore=8,
    )
    body_style = ParagraphStyle(
        "body_style",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
    )

    story = [
        Paragraph("NeuroAI EEG Analyzer - Clinical Report", title_style),
        Paragraph(f"Yaratilgan vaqt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style),
        Spacer(1, 8),
        Paragraph("Bemor ma'lumotlari", section_style),
    ]

    patient_table = Table(
        [
            ["Ism", patient["name"], "ID", patient["patient_id"]],
            ["Yosh", str(patient["age"]), "Jins", patient["gender"]],
            ["Bo'lim", patient["department"], "Texnik", patient["technician"]],
            ["Prioritet", patient["study_priority"], "Manba", recording["source_name"]],
        ],
        colWidths=[62, 160, 62, 170],
    )
    patient_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dff7f5")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.extend([patient_table, Spacer(1, 10), Paragraph("AI xulosa", section_style)])

    summary_table = Table(
        [
            ["Holat", analysis["label"]],
            ["Ishonch", f"{analysis['confidence']:.1f}%"],
            ["Signal sifati", f"{analysis['signal_quality']:.1f}%"],
            ["Artefakt", f"{analysis['artifact_ratio']:.1f}%"],
            ["Seizure risk score", f"{analysis['risk_score']:.1f}/100"],
            ["Dominant kanal", analysis["dominant_channel"]],
        ],
        colWidths=[140, 314],
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#eff6ff"), colors.white]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 8), Paragraph("Klinik izoh", section_style)])
    story.append(Paragraph(patient["clinical_note"], body_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Tizim clinical decision support sifatida ishlatiladi.", body_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Qisqa hisobot", section_style))
    for line in report_text.splitlines()[:20]:
        story.append(Paragraph(line.replace(" ", "&nbsp;"), body_style))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def get_recommended_window_start(events, preview_seconds, total_duration):
    if not events:
        return 0.0
    first_event_center = (events[0]["start"] + events[0]["end"]) / 2
    max_start = max(0.0, total_duration - preview_seconds)
    return float(np.clip(first_event_center - preview_seconds / 2, 0.0, max_start))


def main():
    init_session_state()
    init_database()
    inject_styles()

    if not check_password():
        return

    user = st.session_state["user"]
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-title">NeuroAI EEG Analyzer</div>
            <div class="hero-subtitle">
                EEG signallarini avtomatik qayta ishlash, patologik segmentlarni belgilash,
                risk score va klinik xulosa yaratish uchun mo'ljallangan kengaytirilgan
                klinik platforma prototipi. Kirgan foydalanuvchi: {user['name']} ({user['role']}).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="clinical-note">
            Tizimning interfeysi, bo'limlari, audit log, bemor registri va hisobot oqimi
            klinik ish jarayoniga yaqinlashtirilgan. Lekin yakuniy tashxis shifokor tomonidan beriladi.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("EEG Manbasi")
    source_mode = st.sidebar.radio("Signal turi", ["Demo signal", "Fayl yuklash"])
    st.sidebar.markdown("---")
    st.sidebar.subheader("Preprocessing")
    low_cut = st.sidebar.slider("Low cut (Hz)", 0.1, 5.0, 0.5, 0.1)
    high_cut = st.sidebar.slider("High cut (Hz)", 20.0, 60.0, 40.0, 1.0)
    apply_notch = st.sidebar.checkbox("Notch filter", value=True)
    notch_hz = st.sidebar.selectbox("Notch", [50, 60], index=0)
    st.sidebar.markdown("---")
    preview_seconds = st.sidebar.slider("Preview (sekund)", 6, 30, 12, 2)

    if source_mode == "Demo signal":
        abnormal_demo = st.sidebar.checkbox("Patologik demo", value=True)
        duration_sec = st.sidebar.slider("Davomiylik", 10, 60, 20, 5)
        sfreq = st.sidebar.selectbox("Sampling rate", [128, 256, 512], index=1)
        n_channels = st.sidebar.selectbox("Kanallar", [8, 10, 12], index=2)
        if st.sidebar.button("Demo signal yaratish", use_container_width=True):
            st.session_state.raw_recording = create_demo_signal(
                duration_sec=duration_sec,
                sfreq=sfreq,
                n_channels=n_channels,
                abnormal=abnormal_demo,
            )
            st.session_state.clean_recording = None
            st.session_state.analysis = None
            st.session_state.report_text = ""
            st.session_state.report_pdf = None
            log_audit("load_demo_signal", f"abnormal={abnormal_demo}")
    else:
        uploaded_file = st.sidebar.file_uploader("EEG fayl", type=["edf", "bdf", "csv", "txt"])
        csv_sfreq = st.sidebar.number_input("CSV uchun sampling rate", min_value=64, max_value=4096, value=256)
        if st.sidebar.button("Faylni yuklash", use_container_width=True):
            if uploaded_file is None:
                st.sidebar.error("Avval fayl tanlang.")
            else:
                try:
                    st.session_state.raw_recording = load_recording(uploaded_file, csv_sfreq)
                    st.session_state.clean_recording = None
                    st.session_state.analysis = None
                    st.session_state.report_text = ""
                    st.session_state.report_pdf = None
                    log_audit("load_recording", uploaded_file.name)
                except Exception as exc:
                    st.sidebar.error(str(exc))

    with st.expander("Bemor va tekshiruv ma'lumotlari", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.patient_name = st.text_input("Ism", value=st.session_state.patient_name)
            st.session_state.patient_age = st.number_input("Yosh", min_value=0, max_value=120, value=st.session_state.patient_age)
            gender_options = ["Erkak", "Ayol", "Aniqlanmagan"]
            st.session_state.patient_gender = st.selectbox(
                "Jins",
                gender_options,
                index=gender_options.index(st.session_state.patient_gender),
            )
        with col2:
            st.session_state.patient_id = st.text_input("Bemor ID", value=st.session_state.patient_id)
            department_options = ["Nevrologiya", "Reanimatsiya", "Epileptologiya", "Uyqu laboratoriyasi", "Telemeditsina"]
            st.session_state.department = st.selectbox(
                "Bo'lim",
                department_options,
                index=department_options.index(st.session_state.department),
            )
            priority_options = ["Routine", "Urgent", "ICU Continuous", "Follow-up"]
            st.session_state.study_priority = st.selectbox(
                "Prioritet",
                priority_options,
                index=priority_options.index(st.session_state.study_priority),
            )
        with col3:
            st.session_state.technician = st.text_input("Texnik", value=st.session_state.technician)
            st.session_state.clinical_note = st.text_area("Klinik izoh", value=st.session_state.clinical_note, height=118)

        save_col1, save_col2 = st.columns([1, 3])
        with save_col1:
            if st.button("Bemorni saqlash", use_container_width=True):
                patient_payload = {
                    "name": st.session_state.patient_name,
                    "age": st.session_state.patient_age,
                    "gender": st.session_state.patient_gender,
                    "patient_id": st.session_state.patient_id,
                    "department": st.session_state.department,
                    "clinical_note": st.session_state.clinical_note,
                }
                st.session_state.saved_patient_db_id = save_patient(patient_payload)
                st.success("Bemor registrga saqlandi.")
        with save_col2:
            st.caption("Bemor registri, keyingi EEGlar va audit log platforma ichida lokal SQLite bazada saqlanadi.")

    raw_recording = st.session_state.get("raw_recording")
    if raw_recording is None:
        st.info("Boshlash uchun demo signal yarating yoki EEG fayl yuklang.")
        return

    patient = {
        "name": st.session_state.patient_name,
        "age": st.session_state.patient_age,
        "gender": st.session_state.patient_gender,
        "patient_id": st.session_state.patient_id,
        "department": st.session_state.department,
        "clinical_note": st.session_state.clinical_note,
        "technician": st.session_state.technician,
        "study_priority": st.session_state.study_priority,
    }

    overview1, overview2, overview3, overview4, overview5 = st.columns(5)
    with overview1:
        render_metric_card("Manba", raw_recording["source_name"])
    with overview2:
        render_metric_card("Kanal soni", str(raw_recording["data"].shape[0]))
    with overview3:
        render_metric_card("Sampling rate", f"{raw_recording['sfreq']:.1f} Hz")
    with overview4:
        duration = raw_recording["data"].shape[1] / raw_recording["sfreq"]
        render_metric_card("Davomiylik", f"{duration:.1f} s")
    with overview5:
        render_metric_card("Prioritet", patient["study_priority"])

    if st.button("Klinik tahlilni ishga tushirish", type="primary", use_container_width=True):
        clean_recording = preprocess_recording(raw_recording, low_cut, high_cut, notch_hz, apply_notch)
        analysis = build_analysis(clean_recording, notch_hz)
        report_text = build_report_text(patient, clean_recording, analysis)
        report_pdf = build_report_pdf(patient, clean_recording, analysis, report_text)
        patient_db_id = save_patient(patient)
        study_id = save_study(patient_db_id, clean_recording, analysis, report_text)

        st.session_state.saved_patient_db_id = patient_db_id
        st.session_state.last_saved_study_id = study_id
        st.session_state.clean_recording = clean_recording
        st.session_state.analysis = analysis
        st.session_state.report_text = report_text
        st.session_state.report_pdf = report_pdf
        log_audit("run_analysis", f"study_id={study_id}")

    clean_recording = st.session_state.get("clean_recording")
    analysis = st.session_state.get("analysis")
    if clean_recording is None or analysis is None:
        st.warning("Signal tayyor. Klinik tahlilni ishga tushirish tugmasini bosing.")
        return

    summary1, summary2, summary3, summary4 = st.columns(4)
    with summary1:
        st.metric("AI yorlig'i", analysis["label"])
    with summary2:
        st.metric("Ishonch", f"{analysis['confidence']:.1f}%")
    with summary3:
        st.metric("Signal sifati", f"{analysis['signal_quality']:.1f}%")
    with summary4:
        st.metric("Seizure risk", f"{analysis['risk_score']:.1f}/100")

    dashboard_tab, signal_tab, ai_tab, registry_tab, admin_tab = st.tabs(
        ["Dashboard", "Signal Review", "AI Findings", "Patient Registry", "Reports & Admin"]
    )

    with dashboard_tab:
        st.markdown("<div class=\"section-kicker\">Clinical Overview</div><div class=\"section-title\">Klinik boshqaruv paneli</div>", unsafe_allow_html=True)
        left, middle, right = st.columns([1.1, 1, 1])
        with left:
            render_metric_card("Epileptiform index", f"{analysis['epileptiform_index']:.1f}")
            render_metric_card("Slowing index", f"{analysis['slowing_index']:.2f}")
            render_metric_card("Dominant chastota", f"{analysis['dominant_frequency']:.2f} Hz")
        with middle:
            render_metric_card("Artefakt", f"{analysis['artifact_ratio']:.1f}%")
            render_metric_card("Line noise", f"{analysis['line_noise_ratio']:.1f}%")
            render_metric_card("EMG noise", f"{analysis['emg_ratio']:.1f}%")
        with right:
            render_metric_card("Shubhali segmentlar", str(len(analysis["events"])))
            render_metric_card("Dominant kanal", analysis["dominant_channel"])
            render_metric_card("So'nggi study ID", str(st.session_state.last_saved_study_id or "-"))

        st.info(analysis["recommendation"])
        total_duration = clean_recording["data"].shape[1] / clean_recording["sfreq"]
        recommended_start = get_recommended_window_start(analysis["events"], preview_seconds, total_duration)
        timeline_figure = plot_event_timeline(
            analysis["events"],
            total_duration=total_duration,
            window_start_sec=recommended_start,
            window_duration_sec=preview_seconds,
        )
        st.pyplot(timeline_figure, clear_figure=True)

    with signal_tab:
        st.markdown("<div class=\"section-kicker\">Signal Viewer</div><div class=\"section-title\">Ko'p kanalli ko'rinish va spektrogramma</div>", unsafe_allow_html=True)
        total_duration = clean_recording["data"].shape[1] / clean_recording["sfreq"]
        recommended_start = get_recommended_window_start(analysis["events"], preview_seconds, total_duration)
        max_start = max(0.0, total_duration - preview_seconds)
        control_col1, control_col2, control_col3 = st.columns([1, 1, 1])
        with control_col1:
            window_start_sec = st.number_input(
                "Ko'rish boshlanishi (s)",
                min_value=0.0,
                max_value=float(max_start),
                value=float(recommended_start),
                step=float(max(1.0, preview_seconds / 2)),
            )
        with control_col2:
            channel_limit = int(
                st.selectbox(
                    "Ko'rinadigan kanallar",
                    [4, 6, 8, 10, 12],
                    index=2 if clean_recording["data"].shape[0] >= 8 else 0,
                )
            )
        with control_col3:
            if analysis["events"]:
                event_options = [f"{event['start']:.1f}-{event['end']:.1f} s" for event in analysis["events"][:12]]
                selected_event = st.selectbox("Tez o'tish", ["Tavsiya etilgan oyna"] + event_options)
                if selected_event != "Tavsiya etilgan oyna":
                    chosen_index = event_options.index(selected_event)
                    chosen_event = analysis["events"][chosen_index]
                    window_start_sec = float(max(0.0, min(chosen_event["start"] - preview_seconds / 3, max_start)))
            else:
                st.write("")

        review_figure = plot_signal_review(
            raw_recording,
            clean_recording,
            analysis["events"],
            window_start_sec=window_start_sec,
            preview_seconds=preview_seconds,
            channel_limit=channel_limit,
        )
        st.pyplot(review_figure, clear_figure=True)
        st.caption("Ko'k timeline oynasi va signal review bir xil vaqt oralig'ini ko'rsatadi. To'q sariq zonalar shubhali segmentlar.")

        synced_timeline = plot_event_timeline(
            analysis["events"],
            total_duration=total_duration,
            window_start_sec=window_start_sec,
            window_duration_sec=preview_seconds,
        )
        st.pyplot(synced_timeline, clear_figure=True)

        spec_col1, spec_col2 = st.columns([1, 1])
        with spec_col1:
            selected_channel = st.selectbox("Spektrogramma kanali", clean_recording["channel_names"], index=0)
            spec_figure = plot_spectrogram(
                clean_recording,
                clean_recording["channel_names"].index(selected_channel),
                window_start_sec=window_start_sec,
                window_duration_sec=max(preview_seconds, 20.0),
            )
            st.pyplot(spec_figure, clear_figure=True)
        with spec_col2:
            st.markdown("### Yozuv sifati indikatorlari")
            quality_df = pd.DataFrame(
                [
                    ("Signal sifati", analysis["signal_quality"]),
                    ("Artefakt foizi", analysis["artifact_ratio"]),
                    ("Line noise", analysis["line_noise_ratio"]),
                    ("Baseline drift", analysis["drift_ratio"]),
                    ("EMG noise", analysis["emg_ratio"]),
                ],
                columns=["Ko'rsatkich", "Qiymat"],
            )
            st.dataframe(quality_df, use_container_width=True, hide_index=True)

    with ai_tab:
        st.markdown("<div class=\"section-kicker\">Explainable AI</div><div class=\"section-title\">Model topilmalari va klinik xususiyatlar</div>", unsafe_allow_html=True)
        heatmap_figure = plot_band_heatmap(analysis["band_table"])
        st.pyplot(heatmap_figure, clear_figure=True)

        col1, col2 = st.columns([1.1, 0.9])
        with col1:
            st.markdown("### Ritm ulushlari")
            summary_df = pd.DataFrame(
                [{"Band": band_name, "Mean %": round(analysis["band_means"][band_name], 2)} for band_name in BANDS]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            st.markdown("### Kanal kesimidagi jadval")
            st.dataframe(analysis["band_table"].round(2), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("### Klinik topilmalar")
            st.write(f"AI xulosa: **{analysis['label']}**")
            st.write(f"Seizure probability: **{analysis['seizure_probability']:.1f}%**")
            st.write(f"Dominant kanal: **{analysis['dominant_channel']}**")
            st.write(f"Slowing index: **{analysis['slowing_index']:.2f}**")
            st.write(f"Epileptiform index: **{analysis['epileptiform_index']:.1f}**")
            st.write(f"Tavsiya: **{analysis['recommendation']}**")
            st.markdown("### Segment markerlari")
            if analysis["events"]:
                for event in analysis["events"][:8]:
                    channel_name = clean_recording["channel_names"][event["channel_index"]]
                    st.write(
                        f"- {event['start']:.1f}-{event['end']:.1f} s | {channel_name} | score {event['score']:.2f}"
                    )
            else:
                st.success("Shubhali segment topilmadi.")

    with registry_tab:
        st.markdown("<div class=\"section-kicker\">Clinical Registry</div><div class=\"section-title\">Bemor bazasi va tekshiruvlar tarixi</div>", unsafe_allow_html=True)
        patients_df = load_patients()
        if not patients_df.empty:
            patient_labels = [f"{row.patient_code} - {row.full_name}" for row in patients_df.itertuples(index=False)]
            selected_label = st.selectbox("Registrdan bemor tanlang", patient_labels)
            if st.button("Tanlangan bemorni formaga yuklash", use_container_width=True):
                selected_row = patients_df.iloc[patient_labels.index(selected_label)]
                st.session_state.patient_name = selected_row["full_name"]
                st.session_state.patient_age = int(selected_row["age"]) if pd.notna(selected_row["age"]) else 30
                st.session_state.patient_gender = selected_row["gender"] if pd.notna(selected_row["gender"]) else "Erkak"
                st.session_state.patient_id = selected_row["patient_code"]
                st.session_state.department = selected_row["department"] if pd.notna(selected_row["department"]) else "Nevrologiya"
                st.session_state.clinical_note = selected_row["clinical_note"] if pd.notna(selected_row["clinical_note"]) else ""
                st.rerun()
            st.dataframe(patients_df, use_container_width=True, hide_index=True)
        else:
            st.info("Hali registrda bemor yo'q.")

        studies_df = load_recent_studies()
        st.markdown("### So'nggi EEG tekshiruvlari")
        if not studies_df.empty:
            st.dataframe(studies_df.round(2), use_container_width=True, hide_index=True)
        else:
            st.info("Hali saqlangan study yo'q.")

    with admin_tab:
        st.markdown("<div class=\"section-kicker\">Reports and Governance</div><div class=\"section-title\">Hisobotlar, audit log va platforma holati</div>", unsafe_allow_html=True)
        st.text_area("Avtomatik hisobot", value=st.session_state.report_text, height=360)
        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "TXT hisobotni yuklash",
                data=st.session_state.report_text.encode("utf-8"),
                file_name=f"eeg_report_{patient['patient_id']}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with download_col2:
            st.download_button(
                "PDF hisobotni yuklash",
                data=st.session_state.report_pdf,
                file_name=f"eeg_report_{patient['patient_id']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        admin_col1, admin_col2 = st.columns([1, 1.2])
        with admin_col1:
            render_metric_card("Model versiyasi", MODEL_VERSION)
            render_metric_card("Bemorlar soni", str(len(load_patients())))
            render_metric_card("Study soni", str(len(load_recent_studies(limit=200))))
        with admin_col2:
            st.markdown("### Audit log")
            audit_df = load_audit_logs()
            if not audit_df.empty:
                st.dataframe(audit_df, use_container_width=True, hide_index=True)
            else:
                st.info("Audit log hali bo'sh.")


if __name__ == "__main__":
    main()
