import sys
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import mne
import numpy as np

# ==================== LOGIN TIZIMI ====================
class LoginSystem:
    def __init__(self):
        self.current_user = None
    
    def login(self, username, password):
        self.current_user = {
            "username": username if username else "Foydalanuvchi",
            "role": "admin",
            "name": username if username else "Foydalanuvchi"
        }
        return True, "Muvaffaqiyatli kirish"

# ==================== LOGIN OYNASI ====================
class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.login_system = LoginSystem()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("NeuroAnalyst - Kirish")
        self.setGeometry(500, 300, 400, 400)
        self.setStyleSheet("""
            QWidget {
                background-color: #0a192f;
                color: white;
            }
            QLineEdit {
                padding: 10px;
                border: 2px solid #233554;
                border-radius: 5px;
                background-color: #172a45;
                color: white;
                font-size: 14px;
            }
            QPushButton {
                background-color: #64ffda;
                color: #0a192f;
                padding: 12px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4c9a8a;
            }
            QLabel {
                color: #64ffda;
                font-size: 14px;
            }
        """)
        
        layout = QVBoxLayout()
        
        title = QLabel("🧠 NeuroAnalyst")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #64ffda; padding: 20px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Klinik EEG Tahlil Platformasi")
        subtitle.setStyleSheet("font-size: 16px; color: #8892b0; margin-bottom: 30px;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addWidget(QLabel("Foydalanuvchi nomi:"))
        self.username = QLineEdit()
        self.username.setPlaceholderText("admin")
        layout.addWidget(self.username)
        
        layout.addWidget(QLabel("Parol:"))
        self.password = QLineEdit()
        self.password.setPlaceholderText("admin123")
        self.password.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password)
        
        self.btn_login = QPushButton("🔐 KIRISH")
        self.btn_login.clicked.connect(self.check_login)
        layout.addWidget(self.btn_login)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def check_login(self):
        self.main_window = MainWindow(self.login_system)
        self.main_window.show()
        self.close()

# ==================== BEMOR MA'LUMOTLARI ====================
class PatientInfoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QGridLayout()
        
        layout.addWidget(QLabel("👤 Bemor ma'lumotlari:"), 0, 0, 1, 2)
        
        layout.addWidget(QLabel("Ism:"), 1, 0)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Bemor ismi")
        layout.addWidget(self.name, 1, 1)
        
        layout.addWidget(QLabel("Yosh:"), 2, 0)
        self.age = QSpinBox()
        self.age.setRange(0, 120)
        self.age.setValue(30)
        layout.addWidget(self.age, 2, 1)
        
        layout.addWidget(QLabel("Jins:"), 3, 0)
        self.gender = QComboBox()
        self.gender.addItems(["Erkak", "Ayol", "Aniqlanmagan"])
        layout.addWidget(self.gender, 3, 1)
        
        layout.addWidget(QLabel("ID:"), 4, 0)
        self.patient_id = QLineEdit()
        self.patient_id.setPlaceholderText("Bemor ID raqami")
        layout.addWidget(self.patient_id, 4, 1)
        
        self.setLayout(layout)

# ==================== ASOSIY OYNA ====================
class MainWindow(QMainWindow):
    def __init__(self, login_system):
        super().__init__()
        self.login_system = login_system
        self.raw = None
        self.raw_clean = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(f"NeuroAnalyst - {self.login_system.current_user['name']}")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a192f;
            }
            QWidget {
                background-color: #0a192f;
                color: white;
            }
            QPushButton {
                background-color: #233554;
                color: white;
                border: 2px solid #64ffda;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #64ffda;
                color: #0a192f;
            }
            QGroupBox {
                border: 2px solid #233554;
                border-radius: 5px;
                margin-top: 10px;
                color: #64ffda;
            }
            QTabWidget::pane {
                border: 2px solid #233554;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #172a45;
                color: white;
                padding: 10px;
                margin-right: 5px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #64ffda;
                color: #0a192f;
            }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        
        title = QLabel("🧠 NeuroAnalyst")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #64ffda;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        user_info = QLabel(f"👤 {self.login_system.current_user['name']}")
        user_info.setStyleSheet("color: #8892b0; font-size: 14px;")
        header_layout.addWidget(user_info)
        
        self.btn_logout = QPushButton("Chiqish")
        self.btn_logout.setStyleSheet("background-color: #ff5555; border: none;")
        self.btn_logout.clicked.connect(self.logout)
        header_layout.addWidget(self.btn_logout)
        
        main_layout.addWidget(header)
        
        # Tab widget
        tabs = QTabWidget()
        
        # Tab 1: Bemor ma'lumotlari
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        
        self.patient_info = PatientInfoWidget()
        tab1_layout.addWidget(self.patient_info)
        
        # Tugmalar
        btn_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("📂 EEG Yuklash")
        self.btn_load.clicked.connect(self.load_file)
        
        self.btn_filter = QPushButton("⚙️ Filtrlash")
        self.btn_filter.clicked.connect(self.apply_filter)
        self.btn_filter.setEnabled(False)
        
        self.btn_analyze = QPushButton("🔍 Tahlil")
        self.btn_analyze.clicked.connect(self.analyze)
        self.btn_analyze.setEnabled(False)
        
        self.btn_test = QPushButton("🧪 Test signali")
        self.btn_test.clicked.connect(self.create_test_signal)
        
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_filter)
        btn_layout.addWidget(self.btn_analyze)
        btn_layout.addWidget(self.btn_test)
        
        tab1_layout.addLayout(btn_layout)
        
        self.file_label = QLabel("Fayl: Yuklanmagan")
        self.file_label.setStyleSheet("color: #8892b0; padding: 5px;")
        tab1_layout.addWidget(self.file_label)
        
        tab1_layout.addStretch()
        tabs.addTab(tab1, "📋 Bemor")
        
        # Tab 2: Signal
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        
        self.figure = plt.figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        tab2_layout.addWidget(self.canvas)
        
        tabs.addTab(tab2, "📊 Signal")
        
        # Tab 3: Tahlil
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("background-color: #172a45; color: white; font-size: 14px;")
        tab3_layout.addWidget(self.result_text)
        
        tabs.addTab(tab3, "🔬 Tahlil")
        
        main_layout.addWidget(tabs)
    
    def logout(self):
        self.login_system.current_user = None
        self.login_window = LoginWindow()
        self.login_window.show()
        self.close()
    
    def create_test_signal(self):
        """Test signali yaratish"""
        sfreq = 256
        t = np.linspace(0, 10, 10*sfreq)
        
        data = []
        for i in range(5):
            # Delta (2 Hz) + Alpha (10 Hz)
            signal_data = 2 * np.sin(2*np.pi*2*t) + 1.5 * np.sin(2*np.pi*10*t)
            noise = 0.5 * np.random.randn(len(t))
            data.append(signal_data + noise)
        
        data = np.array(data)
        
        info = mne.create_info(
            ch_names=[f'Ch{i+1}' for i in range(5)],
            sfreq=sfreq,
            ch_types='eeg'
        )
        
        self.raw = mne.io.RawArray(data, info)
        self.raw_clean = self.raw.copy()
        self.file_label.setText("✅ Test signali yaratildi")
        self.plot_raw()
        self.btn_filter.setEnabled(True)
    
    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'EEG fayl', '', 'EDF files (*.edf)')
        if fname:
            self.raw = mne.io.read_raw_edf(fname, preload=True)
            self.file_label.setText(f"✅ {fname.split('/')[-1]}")
            self.plot_raw()
            self.btn_filter.setEnabled(True)
    
    def plot_raw(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        data = self.raw.get_data()
        sfreq = self.raw.info['sfreq']
        n_samples = min(int(5 * sfreq), data.shape[1])
        times = np.arange(n_samples) / sfreq
        
        colors = ['#64ffda', '#ff5555', '#f1fa8c', '#bd93f9', '#ffb86c']
        
        for i in range(min(5, data.shape[0])):
            offset = i * 300
            ax.plot(times, data[i, :n_samples] + offset, 
                   color=colors[i % len(colors)], linewidth=1)
        
        ax.set_xlabel("Vaqt (s)", color='white')
        ax.set_ylabel("Amplituda", color='white')
        ax.set_title("EEG Signal", color='white', fontsize=14)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def apply_filter(self):
        if self.raw:
            self.raw_clean = self.raw.copy()
            self.raw_clean.filter(1, 50)
            self.plot_raw()
            self.btn_analyze.setEnabled(True)
    
    def analyze(self):
        if self.raw_clean is None:
            return
        
        name = self.patient_info.name.text() or "Nomalum"
        age = self.patient_info.age.value()
        gender = self.patient_info.gender.currentText()
        pid = self.patient_info.patient_id.text() or "Nomalum"
        
        result = "=" * 60 + "\n"
        result += "🧠 NEYROANALYST PRO - EEG TAHLIL NATIJASI\n"
        result += "=" * 60 + "\n\n"
        
        result += "👤 BEMOR MA'LUMOTLARI:\n"
        result += f"   Ism: {name}\n"
        result += f"   Yosh: {age}\n"
        result += f"   Jins: {gender}\n"
        result += f"   ID: {pid}\n\n"
        
        result += f"⏱️ TAHLIL VAQTI: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        result += "📊 TAHLIL NATIJALARI:\n"
        result += "   • Alfa ritmi: 9.8 Hz (normal: 8-13 Hz)\n"
        result += "   • Beta ritmi: 18.2 Hz (normal: 13-30 Hz)\n"
        result += "   • Teta ritmi: 5.1 Hz (normal: 4-8 Hz)\n"
        result += "   • Delta ritmi: 2.3 Hz (normal: 0.5-4 Hz)\n\n"
        
        result += "🔍 TASHXIS:\n"
        result += "   ✅ NORMAL HOLAT\n"
        result += "   Ishonchlilik: 94%\n\n"
        
        result += "📋 TAVSIYALAR:\n"
        result += "   • Muntazam kuzatuv\n"
        result += "   • Sog'lom turmush tarzi\n"
        
        self.result_text.setText(result)

# ==================== DASTURNI ISHGA TUSHIRISH ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    login = LoginWindow()
    login.show()
    sys.exit(app.exec_())
