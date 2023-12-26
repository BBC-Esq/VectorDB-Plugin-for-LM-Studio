from functools import partial
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QTextEdit, QTabWidget, QVBoxLayout, QWidget, QGroupBox, QPushButton, QHBoxLayout
from PySide6.QtCore import QUrl
import os
from gui_tabs_settings import GuiSettingsTab
from gui_tabs_tools import GuiSettingsTab as ToolsSettingsTab

def load_url(view, url):
    view.setUrl(QUrl.fromLocalFile(url))

def create_tabs():
    tab_widget = QTabWidget()
    tab_widget.setTabPosition(QTabWidget.South)
    
    tab_font = tab_widget.font()
    tab_font.setPointSize(13)
    tab_widget.setFont(tab_font)

    user_manual_folder = os.path.join(os.path.dirname(__file__), 'User_Manual')

    # SETTINGS TAB
    settings_tab = GuiSettingsTab()
    tab_widget.addTab(settings_tab, 'Settings')

    # USER GUIDE TAB
    user_guide_tab = QWidget()
    user_guide_layout = QVBoxLayout()

    menu_group = QGroupBox("")
    menu_layout = QHBoxLayout()

    buttons_dict = {
        'Usage': 'tips.html',
        'Settings': 'settings.html',
        'Embedding Models': 'embedding_models.html',
        'Whisper': 'transcribe.html'
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
    
    # TOOLS TAB
    tools_tab = ToolsSettingsTab()
    tab_widget.addTab(tools_tab, 'Tools')

    return tab_widget

if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    main_widget = create_tabs()
    main_widget.show()
    sys.exit(app.exec())
