The `QJsonDocument` and `QJsonObject` classes in PySide6 are used for working with JSON documents and objects in Python. Here's a detailed overview of these classes and how you can use them in your Python program.

### QJsonDocument

The `QJsonDocument` class provides a way to read and write JSON documents. It can wrap a complete JSON document and can read this document from, and write it to, a UTF-8 encoded text-based representation.

#### Methods and Functions

- **Constructor (`__init__`)**: Creates a new `QJsonDocument` object.
- **`array()`**: Returns the JSON array contained in the document.
- **`isArray()`**: Returns `True` if the document contains a JSON array.
- **`isEmpty()`**: Returns `True` if the document is empty.
- **`isNull()`**: Returns `True` if the document is null.
- **`isObject()`**: Returns `True` if the document contains a JSON object.
- **`object()`**: Returns the JSON object contained in the document.
- **`__ne__()`**: Inequality operator.
- **`__eq__()`**: Equality operator.
- **`operator[]()`**: Access operator.
- **`setArray(array)`**: Sets the document's content to the given JSON array.
- **`setObject(object)`**: Sets the document's content to the given JSON object.
- **`swap(other)`**: Swaps the contents of this document with another.
- **`toJson()`**: Converts the document to a JSON string.
- **`toVariant()`**: Converts the document to a QVariant.
- **Static Functions**:
  - **`fromJson(const QString &json)`**: Creates a `QJsonDocument` from a JSON string.
  - **`fromVariant(const QVariant &variant)`**: Creates a `QJsonDocument` from a QVariant.

#### Example Usage

```python
from PySide6.QtCore import QJsonDocument, QJsonObject, QByteArray

# Create a JSON object
json_obj = QJsonObject({
    "name": "John",
    "age": 30,
    "city": "New York"
})

# Create a JSON document from the object
json_doc = QJsonDocument(json_obj)

# Convert the document to JSON string
json_str = json_doc.toJson()
print(json_str)

# Check if the document is empty
print(json_doc.isEmpty())

# Convert JSON string back to document
json_doc_from_str = QJsonDocument.fromJson(json_str)

# Check if the new document is valid
if not json_doc_from_str.isNull() and json_doc_from_str.isObject():
    print("Document is valid and contains an object.")
```

### QJsonObject

The `QJsonObject` class encapsulates a JSON object. It can be used to store key-value pairs.

#### Methods

- **Constructor (`__init__`)**: Creates a new `QJsonObject` object.
- **`contains(const QString &key)`**: Returns `True` if the object contains the key.
- **`insert(const QString &key, const QJsonValue &value)`**: Inserts a key-value pair into the object.
- **`keys()`**: Returns a list of keys in the object.
- **`remove(const QString &key)`**: Removes the key and its value from the object.
- **`size()`**: Returns the number of key-value pairs in the object.
- **`value(const QString &key)`**: Returns the value associated with the key.
- **`operator[](const QString &key)`**: Access operator.
- **`swap(QJsonObject &other)`**: Swaps the contents of this object with another.
- **`toVariantMap()`**: Converts the object to a QVariantMap.

#### Example Usage

```python
from PySide6.QtCore import QJsonObject

# Create a JSON object
json_obj = QJsonObject({
    "name": "John",
    "age": 30,
    "city": "New York"
})

# Access a value by key
name = json_obj.value("name").toString()
print(name)

# Check if the object contains a key
if json_obj.contains("age"):
    print("Age is present in the object.")

# Insert a new key-value pair
json_obj.insert("phone", "123-456-7890")

# Remove a key-value pair
json_obj.remove("city")

# Convert to QVariantMap
variant_map = json_obj.toVariantMap()
print(variant_map)
```

### Summary

- **QJsonDocument**: Used for reading and writing JSON documents.
- **QJsonObject**: Encapsulates a JSON object and provides methods to manipulate it.

You can use these classes to handle JSON data in your PySide6 applications. The `QJsonDocument` class provides methods to convert between JSON and Python objects, while `QJsonObject` allows you to manipulate the JSON data as a dictionary.
