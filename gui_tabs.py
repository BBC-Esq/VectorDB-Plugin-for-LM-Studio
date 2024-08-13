from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget, QGroupBox, QPushButton, QHBoxLayout
from pathlib import Path
from functools import partial
from gui_tabs_settings import GuiSettingsTab
from gui_tabs_tools import GuiSettingsTab as ToolsSettingsTab
from gui_tabs_databases import DatabasesTab
from gui_tabs_models import VectorModelsTab
from gui_tabs_database_query import DatabaseQueryTab
from gui_tabs_manage_databases import ManageDatabasesTab

class CustomWebEnginePage(QWebEnginePage):
    def acceptNavigationRequest(self, url, _type, isMainFrame):
        if _type == QWebEnginePage.NavigationTypeLinkClicked:
            QDesktopServices.openUrl(url)
            return False
        return super().acceptNavigationRequest(url, _type, isMainFrame)

def load_url(view, url):
    view.setUrl(QUrl.fromLocalFile(url))

def create_tabs():
    tab_widget = QTabWidget()
    tab_widget.setTabPosition(QTabWidget.South)
    
    tab_font = tab_widget.font()
    tab_font.setPointSize(13)
    tab_widget.setFont(tab_font)
    
    user_manual_folder = Path(__file__).parent / 'User_Manual'
    
    tabs = [
        (GuiSettingsTab(), 'Settings'),
        (create_user_guide_tab(user_manual_folder), 'User Guide'),
        (VectorModelsTab(), 'Models'),
        (ToolsSettingsTab(), 'Tools'),
        (DatabasesTab(), 'Create Database'),
        (ManageDatabasesTab(), 'Manage Databases'),
        (DatabaseQueryTab(), 'Query Database')
    ]
    
    for tab, name in tabs:
        tab_widget.addTab(tab, name)
    
    return tab_widget

def create_user_guide_tab(user_manual_folder):
    user_guide_tab = QWidget()
    user_guide_layout = QVBoxLayout()
    
    user_guide_view = QWebEngineView()
    custom_page = CustomWebEnginePage(user_guide_view)
    user_guide_view.setPage(custom_page)
    user_guide_view.setHtml('<body style="background-color: #161b22;"></body>')
    
    buttons_dict = {
        'Settings': 'settings.html',
        'Embeddings': 'embedding_models.html',
        'Whisper': 'transcribe.html',
        'Vision': 'vision.html',
        'Chat': 'chat.html'
    }
    
    menu_group = QGroupBox("")
    menu_layout = QHBoxLayout()
    
    light_blue_style = "QPushButton { background-color: #3498db; color: white; }"
    
    for button_name, html_file in buttons_dict.items():
        button = QPushButton(button_name)
        button.setStyleSheet(light_blue_style)
        button_url = user_manual_folder / html_file
        button.clicked.connect(partial(load_url, user_guide_view, str(button_url)))
        menu_layout.addWidget(button)
    
    menu_group.setLayout(menu_layout)
    
    user_guide_layout.addWidget(user_guide_view, 9)
    user_guide_layout.addWidget(menu_group, 1)
    user_guide_tab.setLayout(user_guide_layout)
    
    load_url(user_guide_view, str(user_manual_folder / 'settings.html'))
    
    return user_guide_tab