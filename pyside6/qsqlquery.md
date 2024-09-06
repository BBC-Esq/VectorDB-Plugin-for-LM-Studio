The `QSqlQuery` class in PySide6 (the Python bindings for Qt) is used to execute and manipulate SQL statements. It encapsulates the functionality involved in creating, navigating, and retrieving data from SQL queries executed on a `QSqlDatabase`. Here's a detailed overview of `QSqlQuery` and its usage:

### Key Points:

1. **Execution of SQL Statements**:
   - `QSqlQuery` can be used to execute SQL statements such as `SELECT`, `INSERT`, `UPDATE`, and `DELETE`.
   - It can also be used to execute database-specific commands that are not standard SQL (e.g., `SET DATESTYLE=ISO` for PostgreSQL).

2. **Navigating Results**:
   - The class provides methods to navigate through the result set of a query.
   - Methods like `exec()`, `first()`, `next()`, `previous()`, `last()`, `seek()`, etc., are available to manipulate the result set.

3. **Binding Values**:
   - `QSqlQuery` supports binding values to SQL statements. This is done using the `bindValue()` and `addBindValue()` methods.
   - Binding values helps prevent SQL injection and ensures the security of your database operations.

4. **Checking Execution**:
   - The `isActive()` method can be used to check if the query is currently active (executing).

5. **Query Execution**:
   - The `exec()` method is used to execute the SQL statement. It returns a boolean indicating success or failure.

### Example Usage:

Here's a simple example demonstrating how to use `QSqlQuery` to execute a SQL query and retrieve data:

```python
from PySide6.QtSql import QSqlDatabase, QSqlQuery

# Create a database connection
db = QSqlDatabase.addDatabase("QSQLITE")
db.setDatabaseName("example.db")

if not db.open():
    print("Database Error: connection failed")
    return

# Execute a query
query = QSqlQuery()
query.exec("SELECT name, salary FROM employee WHERE salary > 50000")

while query.next():
    name = query.value(0).toString()
    salary = query.value(1).toInt()
    print(f"Name: {name}, Salary: {salary}")

# Close the database connection
db.close()
```

### Methods of `QSqlQuery`:

- **`__init__()`**: Initializes a `QSqlQuery` object.
- **`addBindValue(value, type=None)`**: Adds a value to be bound to a parameter in a prepared query.
- **`bindValue(placeholder, value, type=None)`**: Binds a value to a placeholder in a prepared query.
- **`clear()`**: Clears the state of the query, including any bound values.
- **`exec(query='')`**: Executes the SQL statement specified in the query parameter.
- **`exec_()`**: Alias for `exec()`.
- **`isActive()`**: Returns `True` if the query is currently active.
- **`isForwardOnly()`**: Returns `True` if the query can only be navigated forward through the result set.
- **`next()`**: Moves to the next record in the result set.
- **`value(columnIndex)`**: Retrieves the value of the specified column in the current row.

### Binding Syntax:

- **`QSqlQuery.NamedBinding`**: Uses named placeholders (e.g., `:name`).
- **`QSqlQuery.PositionalBinding`**: Uses positional placeholders (e.g., `?`).

You can set the binding syntax using `setForwardOnly(True)` or `setForwardOnly(False)`.

### Example with Binding:

```python
query = QSqlQuery()
query.prepare("SELECT name, salary FROM employee WHERE salary > :salary")
query.bindValue(":salary", 50000)
query.exec_()

while query.next():
    name = query.value(0).toString()
    salary = query.value(1).toInt()
    print(f"Name: {name}, Salary: {salary}")
```

In summary, `QSqlQuery` is a powerful class for executing and manipulating SQL statements in PySide6, providing a flexible and secure way to interact with databases.
