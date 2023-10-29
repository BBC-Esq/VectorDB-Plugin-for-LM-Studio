from functools import partial
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QTextEdit, QTabWidget, QVBoxLayout, QWidget, QGroupBox, QPushButton, QHBoxLayout, QMessageBox
from PySide6.QtCore import QUrl
import os
from gui_tabs_settings_server import ServerSettingsTab
from gui_tabs_settings_models import ModelsSettingsTab
from gui_tabs_settings_chunks import ChunkSettingsTab  # Import the new ChunkSettingsTab class

def load_url(view, url):
    view.setUrl(QUrl.fromLocalFile(url))

def update_all_configs(server_settings, models_settings, chunk_settings):
    server_updated = server_settings.update_config()
    models_updated = models_settings.update_config()
    chunk_updated = chunk_settings.update_config()

    if server_updated or models_updated or chunk_updated:
        QMessageBox.information(None, 'Settings Updated', 'One or more settings have been updated.')
    else:
        QMessageBox.information(None, 'No Updates', 'No new values were entered.')

def create_tabs():
    tab_widget = QTabWidget()
    tab_widget.setTabPosition(QTabWidget.South)

    user_manual_folder = os.path.join(os.path.dirname(__file__), 'User_Manual')

    # SETTINGS TAB
    settings_tab = QWidget()
    settings_layout = QVBoxLayout()

    # Server settings
    server_settings = ServerSettingsTab()
    server_group = QGroupBox("Server/LLM Settings")
    server_layout = QVBoxLayout()
    server_layout.addWidget(server_settings)
    server_group.setLayout(server_layout)

    # Model settings
    models_settings = ModelsSettingsTab()
    models_group = QGroupBox("Embedding Models Settings")
    models_layout = QVBoxLayout()
    models_layout.addWidget(models_settings)
    models_group.setLayout(models_layout)

    # Chunk settings
    chunk_settings = ChunkSettingsTab()
    chunk_group = QGroupBox("Chunk Settings")
    chunk_layout = QVBoxLayout()
    chunk_layout.addWidget(chunk_settings)
    chunk_group.setLayout(chunk_layout)

    # Update Settings
    update_all_button = QPushButton("Update Settings")
    update_all_button.setFixedWidth(125)
    update_all_button.clicked.connect(lambda: update_all_configs(server_settings, models_settings, chunk_settings))

    # Create a QHBoxLayout for centering the button
    center_button_layout = QHBoxLayout()
    # Add stretches on either side of the button
    center_button_layout.addStretch(1)
    center_button_layout.addWidget(update_all_button)
    center_button_layout.addStretch(1)

    settings_layout.addWidget(server_group, 4)
    settings_layout.addWidget(models_group, 3)
    settings_layout.addWidget(chunk_group, 1)
    
    settings_layout.addLayout(center_button_layout)

    settings_tab.setLayout(settings_layout)
    tab_widget.addTab(settings_tab, 'Settings')

    # DATABASE TAB
    database_tab = QTextEdit("COMING SOON")
    tab_widget.addTab(database_tab, 'Databases')

    # USER GUIDE TAB
    user_guide_tab = QWidget()
    user_guide_layout = QVBoxLayout()

    menu_group = QGroupBox("")
    menu_layout = QHBoxLayout()

    # Dictionary for User Guide buttons
    buttons_dict = {
        'Tips': 'tips.html',
        'Whisper Quants': 'whisper_quants.html',
        'Number Format': 'number_format.html',
        'Settings': 'settings.html'
    }

    # Create WebEngineView
    user_guide_view = QWebEngineView()
    user_guide_view.setHtml('<body style="background-color: #161b22;"></body>')

    # Create User Guide buttons
    for button_name, html_file in buttons_dict.items():
        button = QPushButton(button_name)
        button_url = os.path.join(user_manual_folder, html_file)
        button.clicked.connect(partial(load_url, user_guide_view, button_url))
        menu_layout.addWidget(button)

    menu_group.setLayout(menu_layout)

    user_guide_layout.addWidget(user_guide_view)
    user_guide_layout.addWidget(menu_group)

    # Set stretch factors of User Guide
    user_guide_layout.setStretch(0, 9)
    user_guide_layout.setStretch(1, 1)

    user_guide_tab.setLayout(user_guide_layout)
    tab_widget.addTab(user_guide_tab, 'User Guide')

    # MODELS TAB
    models_tab = QTextEdit("COMING SOON")
    tab_widget.addTab(models_tab, 'Models')

    return tab_widget
