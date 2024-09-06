# QFuture and QFutureWatcher in PySide6

In PySide6 (and the Qt framework), `QFuture` and `QFutureWatcher` are part of the `QtConcurrent` module, which provides high-level APIs for concurrent programming and parallel processing.

## QFuture

`QFuture` represents the result of an asynchronous computation. It allows you to:

- Monitor the progress of the computation
- Retrieve results as they become available
- Check if the computation has finished
- Cancel the computation (if supported)

`QFuture` is a template class, meaning it can hold results of different types.

## QFutureWatcher

`QFutureWatcher` is used to monitor a `QFuture` using Qt's signals and slots mechanism. It provides:

- Signals for various states of the computation (started, finished, canceled, etc.)
- Methods to pause, resume, and cancel the watched operation (if supported)
- A way to get progress information

## Example Usage

Here's an example demonstrating the use of `QFuture` and `QFutureWatcher`:

```python
from PySide6.QtCore import QObject, Slot
from PySide6.QtConcurrent import QtConcurrent
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QProgressBar
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("QFuture and QFutureWatcher Example")
        
        layout = QVBoxLayout()
        
        self.start_button = QPushButton("Start Task")
        self.start_button.clicked.connect(self.start_task)
        layout.addWidget(self.start_button)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    @Slot()
    def start_task(self):
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Create and setup QFutureWatcher
        self.watcher = QtConcurrent.QFutureWatcher()
        self.watcher.finished.connect(self.on_finished)
        self.watcher.progressValueChanged.connect(self.update_progress)
        
        # Start the task
        future = QtConcurrent.run(self.long_running_task)
        self.watcher.setFuture(future)

    def long_running_task(self):
        for i in range(10):
            time.sleep(1)  # Simulate work
            QtConcurrent.reportProgress(i)
        return "Task completed"

    @Slot()
    def on_finished(self):
        result = self.watcher.result()
        print(result)
        self.start_button.setEnabled(True)

    @Slot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value * 10)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
