# tabs.py

from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QTextEdit, QTabWidget
from PySide6.QtCore import QUrl
import os

def create_tabs(tabs_config):
    tab_widget = QTabWidget()
    tab_widget.setTabPosition(QTabWidget.South)

    tab_widgets = [QTextEdit(tab.get('placeholder', '')) for tab in tabs_config]
    for i, tab in enumerate(tabs_config):
        tab_widget.addTab(tab_widgets[i], tab.get('name', ''))

    tutorial_tab = QWebEngineView()
    tab_widget.addTab(tutorial_tab, 'Tutorial')
    user_manual_folder = os.path.join(os.path.dirname(__file__), 'User_Manual')
    html_file_path = os.path.join(user_manual_folder, 'number_format.html')
    tutorial_tab.setUrl(QUrl.fromLocalFile(html_file_path))

    return tab_widget
