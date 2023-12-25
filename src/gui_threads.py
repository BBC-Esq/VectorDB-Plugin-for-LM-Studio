from PySide6.QtCore import QThread, Signal
import server_connector
import create_database

class CreateDatabaseThread(QThread):
    def run(self):
        create_database.main()

class SubmitButtonThread(QThread):
    responseSignal = Signal(str)
    errorSignal = Signal()

    def __init__(self, user_question, parent=None, callback=None):
        super(SubmitButtonThread, self).__init__(parent)
        self.user_question = user_question
        self.callback = callback

    def run(self):
        try:
            response = server_connector.ask_local_chatgpt(self.user_question)
            for response_chunk in response:
                self.responseSignal.emit(response_chunk)
            if self.callback:
                self.callback()
        except Exception as err:
            self.errorSignal.emit()
            print(err)