## Basic Progress Bar Creation

You can create a simple progress bar using the `QProgressBar` widget:

```
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar

app = QApplication([])
window = QWidget()
layout = QVBoxLayout(window)

progress_bar = QProgressBar()
progress_bar.setRange(0, 100)  # Set the range of the progress bar
layout.addWidget(progress_bar)

window.setLayout(layout)
window.show()
app.exec()
```

## Key Methods and Properties

- `setMinimum(minimum)`: Sets the minimum value
- `setMaximum(maximum)`: Sets the maximum value
- `setValue(value)`: Sets the current value
- `setRange(minimum, maximum)`: Sets both minimum and maximum values
- `reset()`: Resets to minimum value
- `setOrientation(orientation)`: Sets orientation (horizontal/vertical)
- `setFormat(format)`: Sets the text format
- `setInvertedAppearance(invert)`: Inverts the bar's appearance
- `setTextVisible(visible)`: Shows/hides the text

Properties: `minimum`, `maximum`, `value`, `orientation`, `format`, `invertedAppearance`, `textVisible`

## QTimer - when a separate thread is not needed

```
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar, QPushButton
from PySide6.QtCore import QTimer

class ProgressBarExample(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        start_button = QPushButton("Start")
        start_button.clicked.connect(self.start_progress)
        layout.addWidget(start_button)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.progress_value = 0

    def start_progress(self):
        self.progress_value = 0
        self.timer.start(100)  # Update every 100ms

    def update_progress(self):
        self.progress_value += 1
        self.progress_bar.setValue(self.progress_value)
        if self.progress_value >= 100:
            self.timer.stop()

app = QApplication([])
window = ProgressBarExample()
window.show()
app.exec()
```

## Qthread - for cpu-intensive tasks

```
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar, QPushButton
from PySide6.QtCore import QThread, Signal, Slot

class Worker(QThread):
    progress = Signal(int)

    def run(self):
        for i in range(101):
            self.progress.emit(i)
            self.msleep(100)  # Simulate work, sleep for 100ms

class ProgressBarExample(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        start_button = QPushButton("Start")
        start_button.clicked.connect(self.start_progress)
        layout.addWidget(start_button)

        self.worker = Worker()
        self.worker.progress.connect(self.update_progress)

    def start_progress(self):
        self.worker.start()

    @Slot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value)

app = QApplication([])
window = ProgressBarExample()
window.show()
app.exec()
```
