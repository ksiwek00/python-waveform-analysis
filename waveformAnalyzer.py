import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt
import pandas as pd
import sys
from PyQt5.QtWidgets import QApplication, QRadioButton, QVBoxLayout, QHBoxLayout, QGridLayout, QTableWidgetItem, QHeaderView, QLabel, QWidget,QFileDialog, QPushButton,QSpinBox, QDoubleSpinBox, QLineEdit, QMessageBox, QAction, QMainWindow, QTableWidget, QTableWidgetItem
import PyQt5.QtGui
import pyqtgraph as pg

def Fourier(t,y):
    N = len(t)
    dt = t[1] -t[0]
    yf = 2.0 / N * np.abs(fft(y)[0:N // 2])
    xf = np.fft.fftfreq(N, d=dt)[0:N // 2]
    return xf, yf

class Generator:
    def __init__(self, min,max,steps):
        self.t = np.linspace(min, max, steps)

    def Sine(self,f,A):
        return A*np.sin(2*np.pi*f*self.t)

    def Square(self,f,A):
        return A*signal.square(2*np.pi*f*self.t)

    def Sawtooth(self,f,A):
        return A*signal.sawtooth(2*np.pi*f*self.t)

    def Triangle(self,f,A):
        return (2 * A)/np.pi * np.arcsin(np.sin(((2*np.pi)/(1/f))*self.t))

    def WhiteNoise(self,A):
        return A * (np.random.rand(len(self.t)) - 0.5)

class App(QMainWindow,QWidget):
    def __init__(self):
        super().__init__()
        # init data
        self.width = 1000
        self.height = 800
        self.left = 100
        self.top = 100

        # geometry
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.window = QWidget()

        # menu
        self.title = "Waveform analysis program"
        self.setWindowTitle(self.title)

        menuBar = self.menuBar()
        sineAction = QAction('Sine', self)
        sineAction.triggered.connect(self.CalculateSine)

        squareAction = QAction('Square', self)
        squareAction.triggered.connect(self.CalculateSquare)

        triangleAction = QAction('Triangle', self)
        triangleAction.triggered.connect(self.CalculateTriangle)

        sawtoothAction = QAction('Sawtooth', self)
        sawtoothAction.triggered.connect(self.CalculateSaw)

        noiseAction = QAction('White Noise', self)
        noiseAction.triggered.connect(self.CalculateNoise)

        fileMenu = menuBar.addMenu('Calculate')
        fileMenu.addAction(sineAction)
        fileMenu.addAction(squareAction)
        fileMenu.addAction(triangleAction)
        fileMenu.addAction(sawtoothAction)
        fileMenu.addAction(noiseAction)

        saveAction = QAction('Save', self)
        saveAction.triggered.connect(self.SaveToFile)
        menuBar.addAction(saveAction)

        ### lewa kolumna

        # labels
        self.labelLayout = QHBoxLayout()

        self.timeLabel = QLabel(self)
        self.timeLabel.setText("Time:")
        self.labelLayout.addWidget(self.timeLabel)
        self.stepsLabel = QLabel(self)
        self.stepsLabel.setText("Steps:")
        self.labelLayout.addWidget(self.stepsLabel)
        self.amplitudeLabel = QLabel(self)
        self.amplitudeLabel.setText("Amplitude:")
        self.labelLayout.addWidget(self.amplitudeLabel)
        self.frequencyLabel = QLabel(self)
        self.frequencyLabel.setText("Frequency:")
        self.labelLayout.addWidget(self.frequencyLabel)

        # inputs
        self.inputsLayout = QHBoxLayout()

        self.time = QDoubleSpinBox(self)
        self.time.setSingleStep(0.1)
        self.time.setRange(0.01, 100)
        self.time.setValue(0.1)
        self.time.valueChanged.connect(self.HandleChange)
        self.inputsLayout.addWidget(self.time)

        self.steps = QSpinBox(self)
        self.steps.setRange(1, 100000)
        self.steps.setValue(20000)
        self.steps.valueChanged.connect(self.HandleChange)
        self.inputsLayout.addWidget(self.steps)

        self.amplitude = QDoubleSpinBox(self)
        self.amplitude.setSingleStep(0.1)
        self.amplitude.setRange(0.01, 2)
        self.amplitude.setValue(1)
        self.amplitude.valueChanged.connect(self.HandleChange)
        self.inputsLayout.addWidget(self.amplitude)

        self.frequency = QDoubleSpinBox(self)
        self.frequency.setSingleStep(0.1)
        self.frequency.setRange(20, 20000)
        self.frequency.setValue(100)
        self.frequency.valueChanged.connect(self.HandleChange)
        self.inputsLayout.addWidget(self.frequency)

        # table
        self.table = QTableWidget(self)

        self.table.resize(400, 375)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(str("Time;Value").split(";"))

        self.header = self.table.horizontalHeader()
        self.header.setSectionResizeMode(0, QHeaderView.Stretch)
        self.header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        ### prawa kolumna

        # radio buttons
        self.radioButtons = []
        self.radioButtons.append(QRadioButton("Sine"))
        self.radioButtons.append(QRadioButton('Square'))
        self.radioButtons.append(QRadioButton('Triangle'))
        self.radioButtons.append(QRadioButton('Sawtooth'))
        self.radioButtons.append(QRadioButton('White noise'))
        self.radioLayout = QVBoxLayout()
        for radio in self.radioButtons:
            self.radioLayout.addWidget(radio)
            radio.toggled.connect(self.HandleChange)

        # wykres 1

        self.graph1 = pg.PlotWidget()
        self.graphPlot1 = self.graph1.plot()
        self.graph1.setLabel('bottom', text='<font size=5>Time (s)</font>')
        self.graph1.setLabel('left', text='<font size=5>Amplitude</font>')
        self.graph1.setTitle('<font size=10>Waveform</font>')

        # wykres 2

        self.graph2 = pg.PlotWidget()
        self.graphPlot2 = self.graph2.plot()
        self.graph2.setLabel('bottom', text='<font size=5>Frequency (Hz)</font>')
        self.graph2.setLabel('left', text='<font size=5>Amplitude</font>')
        self.graph2.setTitle('<font size=10>Fourier transform</font>')

        ### layout
        self.leftCol = QVBoxLayout()
        self.leftCol.addLayout(self.labelLayout)
        self.leftCol.addLayout(self.inputsLayout)
        self.leftCol.addWidget(self.table)

        self.rightCol = QVBoxLayout()
        self.rightCol.addLayout(self.radioLayout)
        self.rightCol.addWidget(self.graph1)
        self.rightCol.addWidget(self.graph2)

        self.mainLayout = QHBoxLayout()
        self.mainLayout.addLayout(self.leftCol)
        self.mainLayout.addLayout(self.rightCol)

        self.window.setLayout(self.mainLayout)
        self.setCentralWidget(self.window)
        self.show()

    def HandleChange(self):
        if self.radioButtons[0].isChecked():
            self.CalculateSine()
        if self.radioButtons[1].isChecked():
            self.CalculateSquare()
        if self.radioButtons[2].isChecked():
            self.CalculateTriangle()
        if self.radioButtons[3].isChecked():
            self.CalculateSaw()
        if self.radioButtons[4].isChecked():
            self.CalculateNoise()

    def CalculateSine(self):
        while (self.table.rowCount() > 0):
            self.table.removeRow(0)
        self.radioButtons[0].setChecked(True)

        generatrorSine = Generator(0, self.time.value(), self.steps.value())
        values = generatrorSine.Sine(self.frequency.value(), self.amplitude.value())
        self.table.setRowCount(self.steps.value())
        for i in range(0,len(generatrorSine.t)):
            self.table.setItem(i, 0, QTableWidgetItem(str(generatrorSine.t[i])))
            self.table.setItem(i, 1, QTableWidgetItem(str(values[i])))

        self.graphPlot1.setData(generatrorSine.t, values)
        self.graphPlot2.setData(Fourier(generatrorSine.t,values)[0], Fourier(generatrorSine.t,values)[1])

    def CalculateSquare(self):
        while (self.table.rowCount() > 0):
            self.table.removeRow(0)
        self.radioButtons[1].setChecked(True)

        generatrorSquare = Generator(0, self.time.value(), self.steps.value())
        values = generatrorSquare.Square(self.frequency.value(), self.amplitude.value())
        self.table.setRowCount(self.steps.value())
        for i in range(0, len(generatrorSquare.t)):
            self.table.setItem(i, 0, QTableWidgetItem(str(generatrorSquare.t[i])))
            self.table.setItem(i, 1, QTableWidgetItem(str(values[i])))

        self.graphPlot1.setData(generatrorSquare.t, values)
        self.graphPlot2.setData(Fourier(generatrorSquare.t,values)[0], Fourier(generatrorSquare.t,values)[1])

    def CalculateTriangle(self):
        while (self.table.rowCount() > 0):
            self.table.removeRow(0)
        self.radioButtons[2].setChecked(True)

        generatrorTriangle = Generator(0, self.time.value(), self.steps.value())
        values = generatrorTriangle.Triangle(self.frequency.value(), self.amplitude.value())
        self.table.setRowCount(self.steps.value())
        for i in range(0, len(generatrorTriangle.t)):
            self.table.setItem(i, 0, QTableWidgetItem(str(generatrorTriangle.t[i])))
            self.table.setItem(i, 1, QTableWidgetItem(str(values[i])))

        self.graphPlot1.setData(generatrorTriangle.t, values)
        self.graphPlot2.setData(Fourier(generatrorTriangle.t,values)[0], Fourier(generatrorTriangle.t,values)[1])

    def CalculateSaw(self):
        while (self.table.rowCount() > 0):
            self.table.removeRow(0)
        self.radioButtons[3].setChecked(True)

        generatrorSaw = Generator(0, self.time.value(), self.steps.value())
        values = generatrorSaw.Sawtooth(self.frequency.value(), self.amplitude.value())
        self.table.setRowCount(self.steps.value())
        for i in range(0, len(generatrorSaw.t)):
            self.table.setItem(i, 0, QTableWidgetItem(str(generatrorSaw.t[i])))
            self.table.setItem(i, 1, QTableWidgetItem(str(values[i])))

        self.graphPlot1.setData(generatrorSaw.t, values)
        self.graphPlot2.setData(Fourier(generatrorSaw.t,values)[0], Fourier(generatrorSaw.t,values)[1])

    def CalculateNoise(self):
        while (self.table.rowCount() > 0):
            self.table.removeRow(0)
        self.radioButtons[4].setChecked(True)

        generatrorNoise = Generator(0, self.time.value(), self.steps.value())
        values = generatrorNoise.WhiteNoise(self.amplitude.value())
        self.table.setRowCount(self.steps.value())
        for i in range(0, len(generatrorNoise.t)):
            self.table.setItem(i, 0, QTableWidgetItem(str(generatrorNoise.t[i])))
            self.table.setItem(i, 1, QTableWidgetItem(str(values[i])))

        self.graphPlot1.setData(generatrorNoise.t, values)
        self.graphPlot2.setData(Fourier(generatrorNoise.t,values)[0], Fourier(generatrorNoise.t,values)[1])

    def SaveToFile(self):
        try:
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getOpenFileName()", "waveFormData.csv", options=options)
            timeData = []
            valueData = []

            for i in range(0,self.table.rowCount()):
                timeData.append(self.table.item(i,0).text())

            for i in range(0, self.table.rowCount()):
                valueData.append(self.table.item(i,1).text())

            data = {"Time": timeData, "Value": valueData}
            dataframe = pd.DataFrame(data)
            dataframe.to_csv(fileName, index=False, sep="\t")
        except:
            print("Save failed")

app = QApplication(sys.argv)
ex = App()
app.exec_()
