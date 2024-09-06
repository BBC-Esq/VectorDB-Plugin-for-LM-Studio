The `QGraphicsItemGroup` class in PySide6 (part of the Qt for Python framework) provides a container that treats a group of items as a single item. This can be useful for managing multiple `QGraphicsItem` objects as a single entity, which can simplify operations like moving, hiding, or transforming them together.

Here's a basic overview of `QGraphicsItemGroup` and its methods:

### Methods

1. **`__init__()`**: Initializes a new `QGraphicsItemGroup` object.
2. **`addToGroup(item)`**: Adds a `QGraphicsItem` to the group.
3. **`removeFromGroup(item)`**: Removes a `QGraphicsItem` from the group.

### Usage Example

Below is an example of how you can use `QGraphicsItemGroup` in a Python program:

```python
import sys
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsItemGroup, QGraphicsRectItem
from PySide6.QtGui import QPen, QColor
from PySide6.QtCore import Qt

class MainWindow(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QGraphicsItemGroup Example")
        self.setSceneRect(-100, -100, 200, 200)
        
        # Create a QGraphicsScene
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Create a QGraphicsItemGroup
        self.group = QGraphicsItemGroup()
        self.scene.addItem(self.group)
        
        # Create a rectangle item
        rect_item = QGraphicsRectItem(0, 0, 50, 50)
        pen = QPen(QColor("red"), 2)
        rect_item.setPen(pen)
        
        # Add the rectangle item to the group
        self.group.addToGroup(rect_item)
        
        # Move the group to the center of the scene
        self.group.setPos(50, 50)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
```

### Explanation

1. **Initialization**:
   - A `QGraphicsScene` is created and set to the `QGraphicsView`.
   - A `QGraphicsItemGroup` is created and added to the scene.

2. **Creating and Adding Items**:
   - A `QGraphicsRectItem` is created with a specified size and color.
   - The rectangle item is added to the `QGraphicsItemGroup` using `addToGroup`.

3. **Positioning**:
   - The `QGraphicsItemGroup` is moved to the center of the scene using `setPos`.

This example demonstrates how to use `QGraphicsItemGroup` to manage multiple items as a single entity, making it easier to manipulate them collectively. You can expand upon this by adding more items to the group and experimenting with different properties and transformations.
