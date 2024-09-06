The `QCalendarWidget` class in PySide6 (the Python bindings for the Qt framework) provides a monthly-based calendar widget that allows users to select a date. It is a versatile widget that can be customized to suit various needs. Here are some key aspects and methods related to `QCalendarWidget`:

### Key Aspects

1. **Properties**:
   - `dateEditAcceptDelay`: The time an inactive date edit is shown before its contents are accepted.
   - `dateEditEnabled`: Whether the date edit popup is enabled.
   - `firstDayOfWeek`: Value identifying the day displayed in the first column.
   - `gridVisible`: Whether the table grid is displayed.
   - `horizontalHeaderFormat`: The format of the horizontal header.
   - `maximumDate`: The maximum date of the currently specified date range.
   - `minimumDate`: The minimum date of the currently specified date range.
   - `navigationBarVisible`: Whether the navigation bar is shown or not.
   - `selectedDate`: The currently selected date.
   - `selectionMode`: The type of selection the user can make in the calendar.

2. **Methods**:
   - `setFirstDayOfWeek(Qt.DayOfWeek day)`: Sets the first day of the week.
   - `setGridVisible(bool visible)`: Sets the visibility of the grid.
   - `setHorizontalHeaderFormat(QCalendarWidget.HorizontalHeaderFormat format)`: Sets the format of the horizontal header.
   - `setMaximumDate(QDate date)`: Sets the maximum date for the calendar.
   - `setMinimumDate(QDate date)`: Sets the minimum date for the calendar.
   - `setNavigationBarVisible(bool visible)`: Sets the visibility of the navigation bar.
   - `setSelectedDate(QDate date)`: Sets the selected date.
   - `setSelectionMode(QCalendarWidget.SelectionMode mode)`: Sets the selection mode of the calendar.

### Example Usage

Here's a simple example demonstrating how to use `QCalendarWidget` in a PySide6 application:

```python
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QCalendarWidget, QVBoxLayout, QWidget, QLabel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a central widget and set it as the main window's central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a layout for the central widget
        layout = QVBoxLayout(central_widget)

        # Create a QCalendarWidget
        calendar = QCalendarWidget()
        layout.addWidget(calendar)

        # Create a label to display the selected date
        self.label = QLabel("Selected Date: ")
        layout.addWidget(self.label)

        # Connect the signal to update the label when a date is selected
        calendar.selectionChanged.connect(self.update_label)

        # Set the main window properties
        self.setWindowTitle("QCalendarWidget Example")
        self.setGeometry(100, 100, 400, 300)

    def update_label(self):
        # Get the selected date from the calendar
        selected_date = self.sender().selectedDate()
        # Update the label with the selected date
        self.label.setText(f"Selected Date: {selected_date.toString()}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

### Explanation

1. **Importing Modules**: Import necessary modules from PySide6.
2. **MainWindow Class**: Define a class `MainWindow` that inherits from `QMainWindow`.
3. **Central Widget and Layout**: Create a central widget and set it as the main window's central widget. Create a layout for the central widget.
4. **QCalendarWidget**: Create a `QCalendarWidget` and add it to the layout.
5. **Label for Displaying Date**: Create a `QLabel` to display the selected date.
6. **Connecting Signals**: Connect the `selectionChanged` signal of the `QCalendarWidget` to the `update_label` method to update the label with the selected date.
7. **Setting Main Window Properties**: Set the window title and geometry.
8. **update_label Method**: This method updates the label with the selected date whenever the date changes in the calendar.

This example demonstrates how to create and use a `QCalendarWidget` in a PySide6 application, along with basic customization and signal handling.
