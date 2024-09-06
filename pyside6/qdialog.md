# QDialog and QProgressDialog in PySide6

PySide6 provides two important classes for creating dialogs and progress feedback: `QDialog` and `QProgressDialog`.

## QDialog

`QDialog` is a top-level window primarily used for short-term tasks and brief communications with the user.

### Key Features:
- Can be modal or modeless
- Provides return values
- Supports default buttons
- Can include a `QSizeGrip` in the lower-right corner

### Types of Dialogs:
1. **Modal**: Blocks interaction with other windows in the application until closed.
2. **Modeless**: Allows interaction with other windows while open.

### Common Use Cases:
- Forms
- Settings panels
- Short-lived tasks

### Example:

```python
from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QPushButton, QLabel

class MyDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My Dialog")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("This is a dialog"))
        self.pushButton = QPushButton("Close", self)
        self.pushButton.clicked.connect(self.accept)
        layout.addWidget(self.pushButton)

app = QApplication([])
dialog = MyDialog()
result = dialog.exec()
print("Dialog accepted" if result == QDialog.Accepted else "Dialog rejected")
