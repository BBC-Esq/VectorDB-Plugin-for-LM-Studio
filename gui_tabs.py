from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QTextEdit, QTabWidget, QVBoxLayout, QWidget, QGroupBox
from PySide6.QtCore import QUrl
import os
from gui_tabs_settings_server import ServerSettingsTab
from gui_tabs_settings_models import ModelsSettingsTab

def create_tabs():
    tab_widget = QTabWidget()
    tab_widget.setTabPosition(QTabWidget.South)

    user_manual_folder = os.path.join(os.path.dirname(__file__), 'User_Manual')

    # Create Settings tab
    settings_tab = QWidget()
    settings_layout = QVBoxLayout()

    # Add Server and EmbeddingModel settings to Settings tab
    server_settings = ServerSettingsTab()
    server_group = QGroupBox("Server/LLM Settings")
    server_layout = QVBoxLayout()
    server_layout.addWidget(server_settings)
    server_group.setLayout(server_layout)

    models_settings = ModelsSettingsTab()
    models_group = QGroupBox("Embedding Models Settings")
    models_layout = QVBoxLayout()
    models_layout.addWidget(models_settings)
    models_group.setLayout(models_layout)

    settings_layout.addWidget(server_group)
    settings_layout.addWidget(models_group)
    settings_tab.setLayout(settings_layout)

    tab_widget.addTab(settings_tab, 'Settings')

    # Create Models tab
    models_tab = QTextEdit("Placeholder text for Models tab.")
    tab_widget.addTab(models_tab, 'Models')

    # Create Floating Point Formats tab
    tutorial_tab = QWebEngineView()
    tutorial_html_path = os.path.join(user_manual_folder, 'number_format.html')
    tutorial_tab.setUrl(QUrl.fromLocalFile(tutorial_html_path))
    tab_widget.addTab(tutorial_tab, 'Floating Point Formats')

    # Create Whisper tab
    whisper_tab = QWebEngineView()
    whisper_html_path = os.path.join(user_manual_folder, 'whisper_quants.html')
    whisper_tab.setUrl(QUrl.fromLocalFile(whisper_html_path))
    tab_widget.addTab(whisper_tab, 'Whisper')
    
    # Create Tips tab
    tips_tab = QWebEngineView()
    tips_html_path = os.path.join(user_manual_folder, 'tips.html')
    tips_tab.setUrl(QUrl.fromLocalFile(tips_html_path))
    tab_widget.addTab(tips_tab, 'Tips')

    return tab_widget
