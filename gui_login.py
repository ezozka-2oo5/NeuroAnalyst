import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MPL_CONFIG_DIR = os.path.join(BASE_DIR, ".mplconfig")
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class LoginSystem:
    def __init__(self):
        self.current_user = None
        self.users = {
            "admin": {"password": "admin123", "name": "Administrator", "role": "admin"},
            "doctor": {"password": "doctor123", "name": "Shifokor", "role": "doctor"},
        }

    def login(self, username, password):
        user = self.users.get(username)
        if user and user["password"] == password:
            self.current_user = {
                "username": username,
                "name": user["name"],
                "role": user["role"],
            }
            return True, "Muvaffaqiyatli kirish"
        return False, "Login yoki parol xato"


class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.login_system = LoginSystem()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("NeuroAnalyst Pro - Kirish")
        self.setFixedSize(520, 520)
        self.setStyleSheet(
            """
            QWidget { background-color: #0a192f; color: white; }
            QLabel { color: #64ffda; font-size: 14px; padding: 4px; }
            QLineEdit {
                padding: 12px;
                border: 2px solid #233554;
                border-radius: 8px;
                background-color: #172a45;
                color: white;
                font-size: 15px;
            }
            QPushButton {
                background-color: #64ffda;
                color: #0a192f;
                padding: 14px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #4c9a8a; color: white; }
            """
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        layout.setContentsMargins(48, 48, 48, 48)

        title = QLabel("NeuroAnalyst Pro")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 30px; font-weight: bold; color: #64ffda;")
        layout.addWidget(title)

        subtitle = QLabel("Klinik EEG Tahlil Platformasi")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 18px; color: #8892b0;")
        layout.addWidget(subtitle)

        layout.addSpacing(18)
        layout.addWidget(QLabel("Foydalanuvchi nomi"))
        self.username = QLineEdit()
        self.username.setPlaceholderText("admin")
        layout.addWidget(self.username)

        layout.addWidget(QLabel("Parol"))
        self.password = QLineEdit()
        self.password.setPlaceholderText("********")
        self.password.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password)

        self.login_button = QPushButton("KIRISH")
        self.login_button.clicked.connect(self.check_login)
        layout.addWidget(self.login_button)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #ff6b6b; font-size: 15px;")
        layout.addWidget(self.status_label)

        demo = QLabel("Demo hisoblar: admin / admin123, doctor / doctor123")
        demo.setWordWrap(True)
        demo.setAlignment(Qt.AlignCenter)
        demo.setStyleSheet("color: #94a3b8; font-size: 13px;")
        layout.addWidget(demo)
        layout.addStretch()

    def check_login(self):
        success, message = self.login_system.login(self.username.text().strip(), self.password.text())
        if success:
            self.main_window = MainWindow(self.login_system)
            self.main_window.show()
            self.close()
        else:
            self.status_label.setText(message)


class PatientInfoWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        layout.setSpacing(14)

        title = QLabel("Bemor ma'lumotlari")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #64ffda;")
        layout.addWidget(title, 0, 0, 1, 2)

        layout.addWidget(QLabel("Ism"), 1, 0)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Bemor ismi")
        layout.addWidget(self.name, 1, 1)

        layout.addWidget(QLabel("Yosh"), 2, 0)
        self.age = QSpinBox()
        self.age.setRange(0, 120)
        self.age.setValue(30)
        layout.addWidget(self.age, 2, 1)

        layout.addWidget(QLabel("Jins"), 3, 0)
        self.gender = QComboBox()
        self.gender.addItems(["Erkak", "Ayol"])
        layout.addWidget(self.gender, 3, 1)

        layout.addWidget(QLabel("ID"), 4, 0)
        self.patient_id = QLineEdit()
        self.patient_id.setPlaceholderText("Bemor ID")
        layout.addWidget(self.patient_id, 4, 1)


class AgeNormalizer:
    def __init__(self):
        self.normal_ranges = {
            "0-2": {"delta": (2.0, 4.0), "theta": (4.0, 6.0), "alpha": (6.0, 8.0), "beta": (8.0, 12.0)},
            "3-5": {"delta": (1.5, 3.5), "theta": (3.5, 5.5), "alpha": (5.5, 9.0), "beta": (9.0, 13.0)},
            "6-12": {"delta": (1.0, 3.0), "theta": (3.0, 5.0), "alpha": (5.0, 10.0), "beta": (10.0, 15.0)},
            "13-20": {"delta": (0.5, 2.5), "theta": (2.5, 4.5), "alpha": (8.0, 12.0), "beta": (12.0, 18.0)},
            "21-50": {"delta": (0.5, 2.0), "theta": (2.0, 4.0), "alpha": (8.5, 12.5), "beta": (12.5, 25.0)},
            "51+": {"delta": (0.5, 2.0), "theta": (2.0, 4.0), "alpha": (7.5, 11.5), "beta": (11.5, 22.0)},
        }

    def get_age_group(self, age):
        if age <= 2:
            return "0-2"
        if age <= 5:
            return "3-5"
        if age <= 12:
            return "6-12"
        if age <= 20:
            return "13-20"
        if age <= 50:
            return "21-50"
        return "51+"

    def analyze(self, age, values):
        age_group = self.get_age_group(age)
        normal = self.normal_ranges[age_group]
        results = {}
        for band, value in values.items():
            low, high = normal[band]
            results[band] = {
                "value": value,
                "normal_range": f"{low}-{high}",
                "status": "normal" if low <= value <= high else "abnormal",
            }
        return results


class AIDiagnosis:
    def diagnose(self, results):
        abnormal_count = sum(1 for result in results.values() if result["status"] == "abnormal")
        if abnormal_count == 0:
            return {
                "diagnosis": "NORMAL HOLAT",
                "confidence": 95,
                "recommendations": [
                    "Muntazam kuzatuv",
                    "Sog'lom turmush tarzi",
                    "Yiliga 1 marta profilaktik tekshiruv",
                ],
            }
        if abnormal_count <= 2:
            return {
                "diagnosis": "CHEGARAVIY HOLAT",
                "confidence": 82,
                "recommendations": [
                    "Nevrolog konsultatsiyasi",
                    "Qo'shimcha tekshiruvlar",
                    "3 oydan keyin qayta EEG",
                ],
            }
        return {
            "diagnosis": "PATOLOGIK O'ZGARISHLAR",
            "confidence": 91,
            "recommendations": [
                "Shoshilinch nevrolog konsultatsiyasi",
                "Video-EEG monitoring",
                "Antiepileptik terapiya maslahati",
                "1 oydan keyin qayta tekshiruv",
            ],
        }


class MainWindow(QMainWindow):
    def __init__(self, login_system):
        super().__init__()
        self.login_system = login_system
        self.raw_data = None
        self.filtered_data = None
        self.age_normalizer = AgeNormalizer()
        self.ai_diagnosis = AIDiagnosis()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"NeuroAnalyst Pro - {self.login_system.current_user['name']}")
        self.setGeometry(100, 100, 1360, 860)
        self.setStyleSheet(
            """
            QMainWindow { background-color: #0a192f; }
            QWidget { background-color: #0a192f; color: white; font-family: Segoe UI; }
            QPushButton {
                background-color: #233554;
                color: white;
                border: 2px solid #64ffda;
                padding: 12px 18px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover { background-color: #64ffda; color: #0a192f; }
            QTabWidget::pane { border: 2px solid #233554; border-radius: 8px; }
            QTabBar::tab {
                background-color: #172a45;
                color: white;
                padding: 12px 22px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected { background-color: #64ffda; color: #0a192f; }
            QLineEdit, QSpinBox, QComboBox, QTextEdit {
                background-color: #172a45;
                color: white;
                border: 2px solid #233554;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }
            QLabel { color: #cbd5e1; font-size: 14px; }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        self.build_header(root)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.build_patient_tab()
        self.build_signal_tab()
        self.build_analysis_tab()

    def build_header(self, layout):
        header = QWidget()
        row = QHBoxLayout(header)

        title = QLabel("NeuroAnalyst Pro")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #64ffda;")
        row.addWidget(title)
        row.addStretch()

        user = self.login_system.current_user
        badge = QLabel(f"{user['name']} ({user['role']})")
        badge.setStyleSheet(
            "background-color: #172a45; color: #64ffda; padding: 8px 14px; border-radius: 16px;"
        )
        row.addWidget(badge)

        logout = QPushButton("Chiqish")
        logout.setStyleSheet("QPushButton { background-color: #ef4444; border: none; } QPushButton:hover { background-color: #dc2626; color: white; }")
        logout.clicked.connect(self.logout)
        row.addWidget(logout)
        layout.addWidget(header)

    def build_patient_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        self.patient_info = PatientInfoWidget()
        layout.addWidget(self.patient_info)

        button_row = QHBoxLayout()
        self.generate_button = QPushButton("Test ma'lumot")
        self.generate_button.clicked.connect(self.generate_test_data)
        button_row.addWidget(self.generate_button)

        self.load_button = QPushButton("EEG fayl yuklash")
        self.load_button.clicked.connect(self.load_file)
        button_row.addWidget(self.load_button)

        self.filter_button = QPushButton("Filtrlash")
        self.filter_button.clicked.connect(self.filter_signal)
        self.filter_button.setEnabled(False)
        button_row.addWidget(self.filter_button)

        button_row.addStretch()
        layout.addLayout(button_row)

        self.status_label = QLabel("Holat: bemor ma'lumotlarini kiriting yoki test signal yarating.")
        self.status_label.setStyleSheet(
            "background-color: #172a45; color: #64ffda; padding: 12px; border-radius: 8px;"
        )
        layout.addWidget(self.status_label)
        layout.addStretch()

        self.tabs.addTab(tab, "Bemor")

    def build_signal_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)

        self.figure = Figure(figsize=(13, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.tabs.addTab(tab, "Signal")

    def build_analysis_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        self.analyze_button = QPushButton("Tahlil qilish")
        self.analyze_button.clicked.connect(self.analyze)
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet(
            "QPushButton { background-color: #64ffda; color: #0a192f; font-size: 17px; padding: 16px; }"
        )
        layout.addWidget(self.analyze_button)

        self.tabs.addTab(tab, "Tahlil")

    def generate_test_data(self):
        self.patient_info.name.setText("Test Bemor")
        self.patient_info.age.setValue(int(np.random.randint(20, 60)))
        self.patient_info.gender.setCurrentIndex(int(np.random.randint(0, 2)))
        self.patient_info.patient_id.setText(f"TEST{int(np.random.randint(1000, 9999))}")
        self.status_label.setText("Holat: test ma'lumotlari yaratildi.")
        self.create_test_signal()
        self.filter_button.setEnabled(True)

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "EEG faylini tanlang", "", "EDF files (*.edf)")
        if not file_name:
            return
        self.status_label.setText(f"Holat: {file_name.split('/')[-1]} yuklandi. Demo signal tayyorlandi.")
        self.create_test_signal()
        self.filter_button.setEnabled(True)

    def create_test_signal(self):
        sample_rate = 250
        times = np.linspace(0, 10, 10 * sample_rate)
        signals = []

        for index in range(5):
            base_signal = (
                0.5 * np.sin(2 * np.pi * 2 * times)
                + 0.3 * np.sin(2 * np.pi * 6 * times)
                + 0.4 * np.sin(2 * np.pi * 10 * times)
                + 0.2 * np.sin(2 * np.pi * 20 * times)
            )
            noise = 0.1 * np.random.randn(len(times))
            if index in (1, 3):
                base_signal += np.exp(-((times - 6.0) / 0.05) ** 2) * 2.5
            signals.append(base_signal + noise)

        self.raw_data = np.asarray(signals)
        self.filtered_data = None
        self.analyze_button.setEnabled(False)
        self.plot_signals(filtered=False)

    def filter_signal(self):
        if self.raw_data is None:
            return

        filtered_channels = []
        kernel = np.ones(10) / 10
        for channel in self.raw_data:
            filtered_channels.append(np.convolve(channel, kernel, mode="same"))

        self.filtered_data = np.asarray(filtered_channels)
        self.plot_signals(filtered=True)
        self.analyze_button.setEnabled(True)
        self.status_label.setText("Holat: signal filtrlandi va tahlilga tayyor.")

    def plot_signals(self, filtered):
        self.figure.clear()
        raw_axis = self.figure.add_subplot(2, 1, 1)
        filtered_axis = self.figure.add_subplot(2, 1, 2)

        sample_rate = 250
        times = np.linspace(0, 10, 10 * sample_rate)

        if self.raw_data is not None:
            for index, channel in enumerate(self.raw_data[:5]):
                raw_axis.plot(times, channel + index * 5, linewidth=0.9, alpha=0.85)

        raw_axis.set_title("Xom signal", color="white")
        raw_axis.set_xlabel("Vaqt (s)", color="white")
        raw_axis.set_ylabel("Amplituda", color="white")
        raw_axis.tick_params(colors="white")
        raw_axis.grid(True, alpha=0.2)
        raw_axis.set_facecolor("#172a45")

        if filtered and self.filtered_data is not None:
            for index, channel in enumerate(self.filtered_data[:5]):
                filtered_axis.plot(times, channel + index * 5, linewidth=0.9, alpha=0.9, color="#64ffda")

        filtered_axis.set_title("Filtrlangan signal", color="white")
        filtered_axis.set_xlabel("Vaqt (s)", color="white")
        filtered_axis.set_ylabel("Amplituda", color="white")
        filtered_axis.tick_params(colors="white")
        filtered_axis.grid(True, alpha=0.2)
        filtered_axis.set_facecolor("#172a45")

        self.figure.patch.set_facecolor("#0a192f")
        self.figure.tight_layout()
        self.canvas.draw()

    def analyze(self):
        if self.filtered_data is None:
            return

        patient_name = self.patient_info.name.text() or "Nomalum"
        age = self.patient_info.age.value()
        gender = self.patient_info.gender.currentText()
        patient_id = self.patient_info.patient_id.text() or "Nomalum"

        first_channel = np.abs(self.filtered_data[0])
        values = {
            "delta": float(np.mean(first_channel) * 2.5),
            "theta": float(np.mean(first_channel) * 1.8),
            "alpha": float(np.mean(first_channel) * 2.2),
            "beta": float(np.mean(first_channel) * 1.5),
        }

        results = self.age_normalizer.analyze(age, values)
        diagnosis = self.ai_diagnosis.diagnose(results)

        lines = [
            "=" * 78,
            "NEUROANALYST PRO - KLINIK EEG TAHLILI".center(78),
            "=" * 78,
            "",
            "BEMOR MA'LUMOTLARI",
            f"Ism: {patient_name}",
            f"Yosh: {age}",
            f"Jins: {gender}",
            f"ID: {patient_id}",
            f"Vaqt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Shifokor: {self.login_system.current_user['name']}",
            "",
            "SIGNAL XUSUSIYATLARI",
        ]

        for band, result in results.items():
            status = "OK" if result["status"] == "normal" else "ABNORMAL"
            lines.append(
                f"{status}  {band.upper()}: {result['value']:.2f} Hz (norma: {result['normal_range']})"
            )

        lines.extend(
            [
                "",
                "SUN'IY INTELLEKT TASHXISI",
                f"Holat: {diagnosis['diagnosis']}",
                f"Ishonchlilik: {diagnosis['confidence']}%",
                "",
                "TAVSIYALAR",
            ]
        )

        for recommendation in diagnosis["recommendations"]:
            lines.append(f"- {recommendation}")

        self.result_text.setText("\n".join(lines))
        self.tabs.setCurrentIndex(2)

    def logout(self):
        self.login_system.current_user = None
        self.login_window = LoginWindow()
        self.login_window.show()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(10, 25, 47))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(23, 42, 69))
    palette.setColor(QPalette.AlternateBase, QColor(23, 42, 69))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(35, 53, 84))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(100, 255, 218))
    palette.setColor(QPalette.HighlightedText, QColor(10, 25, 47))
    app.setPalette(palette)

    login = LoginWindow()
    login.show()
    sys.exit(app.exec_())
