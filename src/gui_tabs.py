from functools import partial
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QTextEdit, QTabWidget, QVBoxLayout, QWidget, QGroupBox, QPushButton, QHBoxLayout, QMessageBox
from PySide6.QtCore import QUrl
import os
from gui_tabs_settings import GuiSettingsTab

def load_url(view, url):
    view.setUrl(QUrl.fromLocalFile(url))

def create_tabs():
    tab_widget = QTabWidget()
    tab_widget.setTabPosition(QTabWidget.South)

    user_manual_folder = os.path.join(os.path.dirname(__file__), 'User_Manual')

    # SETTINGS TAB
    settings_tab = GuiSettingsTab()
    tab_widget.addTab(settings_tab, 'Settings')

    # DATABASE TAB
    database_tab = QTextEdit("COMING SOON")
    tab_widget.addTab(database_tab, 'Databases')

    # USER GUIDE TAB
    user_guide_tab = QWidget()
    user_guide_layout = QVBoxLayout()

    menu_group = QGroupBox("")
    menu_layout = QHBoxLayout()

    buttons_dict = {
        'Tips': 'tips.html',
        'Whisper Quants': 'whisper_quants.html',
        'Number Format': 'number_format.html',
        'Settings': 'settings.html'
    }

    user_guide_view = QWebEngineView()
    user_guide_view.setHtml('<body style="background-color: #161b22;"></body>')

    for button_name, html_file in buttons_dict.items():
        button = QPushButton(button_name)
        button_url = os.path.join(user_manual_folder, html_file)
        button.clicked.connect(partial(load_url, user_guide_view, button_url))
        menu_layout.addWidget(button)

    menu_group.setLayout(menu_layout)
    user_guide_layout.addWidget(user_guide_view)
    user_guide_layout.addWidget(menu_group)
    user_guide_layout.setStretch(0, 9)
    user_guide_layout.setStretch(1, 1)

    user_guide_tab.setLayout(user_guide_layout)
    tab_widget.addTab(user_guide_tab, 'User Guide')

    # MODELS TAB
    models_tab = QTextEdit("COMING SOON")
    tab_widget.addTab(models_tab, 'Models')
    
    # TOOLS TAB
    # tools_tab = ToolsTab()
    # tab_widget.addTab(tools_tab, 'Settings')

    return tab_widget
