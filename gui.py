import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("EEG Test")
        self.setGeometry(200, 200, 800, 600)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Tugma
        self.btn = QPushButton("📊 Sinov signalini chizish")
        self.btn.clicked.connect(self.plot_test)
        layout.addWidget(self.btn)
        
        # Grafik
        self.figure = plt.figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
    def plot_test(self):
        """SODDA SINOV SIGNALI - bu albatta ishlaydi"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Vaqt o'qi (0 dan 10 soniyagacha)
        t = np.linspace(0, 10, 1000)
        
        # Sinov signallari - to'lqinlar
        for i in range(5):
            # Turli chastotali to'lqinlar
            signal = np.sin(2 * np.pi * (i+1) * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
            # Har bir kanalga ofset qo'shish
            ax.plot(t, signal + i*3, linewidth=1, label=f'Kanal {i+1}')
        
        ax.set_xlabel("Vaqt (soniya)")
        ax.set_ylabel("Amplituda")
        ax.set_title("SINOV SIGNALI - To'lqinlar")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())python gui.py