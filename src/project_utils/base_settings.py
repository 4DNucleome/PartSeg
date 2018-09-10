"""from PyQt5.QtCore import pyqtSignal, QObject
import sqlite3
import os

class BaseSettings(QObject):
    def __init__(self, database_path):
        super().__init__()
        self.database = sqlite3.connect(database_path)
        self.database.row_factory = sqlite3.Row
        if not os.path.exists(database_path):
            self.create_database(database_path)


    def create_database(self, database_path):

        pass"""