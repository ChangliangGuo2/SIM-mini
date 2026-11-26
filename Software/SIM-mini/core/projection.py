import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .constants import PROJECT_W, PROJECT_H
from .image_io import rgb_to_qimage


class ProjectionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pattern Projection")
        self.setStyleSheet("background: black;")

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background: black;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

        self.current_qimage = None
        self.test_mode = False
        self._screen = None

    def open_on_screen(self, screen, test_mode: bool = False):
        self._screen = screen
        self.test_mode = bool(test_mode)
        self.hide()

        geo = screen.geometry()
        available = screen.availableGeometry()

        if self.test_mode:
            self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
            self.setFixedSize(PROJECT_W, PROJECT_H)
            x = available.x() + max(0, (available.width() - PROJECT_W) // 2)
            y = available.y() + max(0, (available.height() - PROJECT_H) // 2)
            self.setGeometry(x, y, PROJECT_W, PROJECT_H)
        else:
            self.setMinimumSize(0, 0)
            self.setMaximumSize(16777215, 16777215)
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.setGeometry(geo)

        self.winId()
        if self.windowHandle() is not None:
            self.windowHandle().setScreen(screen)

        if self.test_mode:
            self.show()
        else:
            self.showFullScreen()
            self.setGeometry(geo)

        self.show_black()

    def ensure_open(self):
        """Reopen the projection window if the user closed it.

        In test mode the projection is intentionally a 640x480 popup on the
        secondary screen.  Closing the popup should not make the next pattern
        silently disappear; the next show_* call reopens it at the correct size.
        """
        if self._screen is None:
            return False

        needs_reopen = not self.isVisible()
        if self.test_mode and (self.width() != PROJECT_W or self.height() != PROJECT_H):
            needs_reopen = True

        if needs_reopen:
            self.open_on_screen(self._screen, test_mode=self.test_mode)
        return True

    def show_black(self):
        black = np.zeros((PROJECT_H, PROJECT_W, 3), dtype=np.uint8)
        self.show_pattern(black)

    def show_white(self):
        white = np.full((PROJECT_H, PROJECT_W, 3), 255, dtype=np.uint8)
        self.show_pattern(white)

    def show_pattern(self, image_rgb: np.ndarray):
        self.ensure_open()
        self.current_qimage = rgb_to_qimage(image_rgb)
        self._refresh_pixmap()

    def closeEvent(self, event):
        # Keep the object reusable; show_pattern()/show_white() will reopen it.
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if self.current_qimage is None:
            return

        pix = QPixmap.fromImage(self.current_qimage)
        scale_mode = Qt.KeepAspectRatio if self.test_mode else Qt.IgnoreAspectRatio

        target_size = self.label.size() if self.label.size().isValid() else self.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            return

        self.label.setPixmap(
            pix.scaled(
                target_size,
                scale_mode,
                Qt.FastTransformation,
            )
        )
