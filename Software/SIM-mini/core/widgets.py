from PySide6.QtCore import Qt, QPoint, Signal, QSize, QTimer, QRect
from PySide6.QtGui import QPixmap, QPainter, QPainterPath, QColor, QPen, QBrush
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from .image_io import rgb_to_qimage


class ImageResultDialog(QDialog):
    def __init__(self, image_rgb, title="Result", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1500, 560)
        self.setMinimumSize(640, 360)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(320, 240)
        self.label.setStyleSheet(
            "background: #202020; color: #dddddd; border: 1px solid #555;"
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

        self.qimage = None
        self.pixmap = None
        self.set_image(image_rgb)

    def set_image(self, image_rgb):
        self.qimage = rgb_to_qimage(image_rgb)
        self.pixmap = QPixmap.fromImage(self.qimage)
        self.update_pixmap()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.update_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap()

    def update_pixmap(self):
        if self.pixmap is None:
            return
        target_size = self.label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            target_size = self.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            return
        self.label.setPixmap(
            self.pixmap.scaled(
                target_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class ReviewImageLabel(QLabel):
    imageClicked = Signal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_size = QSize()
        self._display_rect = QRect()

    def set_source_size(self, size: QSize):
        self._source_size = QSize(size)

    def set_display_rect(self, rect: QRect):
        self._display_rect = QRect(rect)

    def mousePressEvent(self, event):
        if self._source_size.width() <= 0 or self._source_size.height() <= 0:
            return super().mousePressEvent(event)

        pos = event.position().toPoint()
        rect = self._display_rect if not self._display_rect.isNull() else self.rect()
        if not rect.contains(pos):
            return super().mousePressEvent(event)

        x = int(round((pos.x() - rect.x()) * self._source_size.width() / max(1, rect.width())))
        y = int(round((pos.y() - rect.y()) * self._source_size.height() / max(1, rect.height())))
        x = max(0, min(self._source_size.width() - 1, x))
        y = max(0, min(self._source_size.height() - 1, y))
        button = event.button()
        if hasattr(button, "value"):
            button_value = int(button.value)
        else:
            button_value = int(button)
        self.imageClicked.emit(x, y, button_value)


class MaskReviewDialog(QDialog):
    imageClicked = Signal(int, int, int)

    """Resizable Mask preview window with editable parameter panel.

    The mode widgets own the actual parameter controls on the main UI.  This
    dialog receives cloned controls from the mode and exposes buttons so the
    mode can apply/recalculate/save the current mask from one place.
    """

    def __init__(self, image_rgb, title="Mask Result", params_widget=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1560, 620)
        self.setMinimumSize(820, 420)

        self.label = ReviewImageLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(480, 320)
        self.label.setStyleSheet(
            "background: #202020; color: #dddddd; border: 1px solid #555;"
        )
        self.label.imageClicked.connect(self.imageClicked.emit)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.label)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.scroll_area)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        if params_widget is not None:
            right_layout.addWidget(params_widget)
        right_layout.addStretch(1)

        self.recalculate_btn = QPushButton("Apply Parameters && Recalculate")
        self.save_btn = QPushButton("Save Mask && Apply Parameters")
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)

        right_layout.addWidget(self.recalculate_btn)
        right_layout.addWidget(self.save_btn)
        right_layout.addWidget(self.close_btn)

        layout = QHBoxLayout(self)
        layout.addLayout(left_layout, stretch=1)
        layout.addWidget(right_panel, stretch=0)

        self.qimage = None
        self.pixmap = None
        self.set_image(image_rgb)

    def set_image(self, image_rgb):
        self.qimage = rgb_to_qimage(image_rgb)
        self.pixmap = QPixmap.fromImage(self.qimage)
        if hasattr(self.label, "set_source_size"):
            self.label.set_source_size(self.pixmap.size())
        self.update_pixmap()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.update_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap()

    def update_pixmap(self):
        if self.pixmap is None:
            return
        target_size = self.label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            target_size = self.scroll_area.viewport().size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            return
        scaled = self.pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.label.setPixmap(scaled)
        if hasattr(self.label, "set_display_rect"):
            x = max(0, (self.label.width() - scaled.width()) // 2)
            y = max(0, (self.label.height() - scaled.height()) // 2)
            self.label.set_display_rect(QRect(x, y, scaled.width(), scaled.height()))


class HistogramContrastWidget(QWidget):
    levelsChanged = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(150)
        self.setMinimumWidth(300)
        self.setMouseTracking(True)

        self.histogram = np.zeros(256, dtype=np.float32)
        self.black_level = 0
        self.white_level = 255
        self.draggable = True
        self._drag_target = None

    def sizeHint(self):
        return QSize(360, 150)

    def set_histogram(self, hist):
        hist = np.asarray(hist, dtype=np.float32)
        if hist.size == 256:
            self.histogram = hist.copy()
            self.update()

    def set_levels(self, black_level: int, white_level: int, emit_signal: bool = False):
        black_level = int(max(0, min(254, black_level)))
        white_level = int(max(black_level + 1, min(255, white_level)))

        changed = black_level != self.black_level or white_level != self.white_level

        self.black_level = black_level
        self.white_level = white_level
        self.update()

        if changed and emit_signal:
            self.levelsChanged.emit(self.black_level, self.white_level)

    def set_draggable(self, enabled: bool):
        self.draggable = bool(enabled)
        self.setCursor(Qt.ArrowCursor if enabled else Qt.ForbiddenCursor)

    def _plot_rect(self):
        return self.rect().adjusted(10, 10, -10, -24)

    def _level_to_x(self, level: int) -> int:
        r = self._plot_rect()
        return int(r.left() + (level / 255.0) * r.width())

    def _x_to_level(self, x: int) -> int:
        r = self._plot_rect()
        t = (x - r.left()) / max(1, r.width())
        return int(max(0, min(255, round(t * 255))))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        outer = self.rect().adjusted(0, 0, -1, -1)
        plot = self._plot_rect()

        painter.fillRect(outer, QColor(28, 28, 28))
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        painter.drawRoundedRect(outer, 6, 6)

        painter.fillRect(plot, QColor(18, 18, 18))
        painter.setPen(QPen(QColor(48, 48, 48), 1))
        for i in range(1, 4):
            x = plot.left() + int(plot.width() * i / 4)
            painter.drawLine(x, plot.top(), x, plot.bottom())
        for i in range(1, 3):
            y = plot.top() + int(plot.height() * i / 3)
            painter.drawLine(plot.left(), y, plot.right(), y)

        if np.max(self.histogram) > 0:
            hist = self.histogram.astype(np.float32)
            log_hist = np.log1p(hist)
            log_hist /= max(1e-6, float(np.max(log_hist)))

            path = QPainterPath()
            for i in range(256):
                x = plot.left() + (i / 255.0) * plot.width()
                y = plot.bottom() - log_hist[i] * plot.height()
                if i == 0:
                    path.moveTo(x, plot.bottom())
                    path.lineTo(x, y)
                else:
                    path.lineTo(x, y)
            path.lineTo(plot.right(), plot.bottom())
            path.closeSubpath()

            painter.setBrush(QBrush(QColor(80, 170, 255, 100)))
            painter.setPen(Qt.NoPen)
            painter.drawPath(path)

            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(120, 210, 255), 2))
            painter.drawPath(path)

        self._draw_level_line(
            painter,
            self.black_level,
            QColor(80, 170, 255),
            f"B {self.black_level}",
        )
        self._draw_level_line(
            painter,
            self.white_level,
            QColor(255, 170, 60),
            f"W {self.white_level}",
        )

        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.drawText(10, self.height() - 7, "Drag blue/orange lines to adjust display contrast")

    def _draw_level_line(self, painter, level: int, color: QColor, text: str):
        plot = self._plot_rect()
        x = self._level_to_x(level)

        painter.setPen(QPen(color, 2))
        painter.drawLine(x, plot.top(), x, plot.bottom())

        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(
            [
                QPoint(x, plot.top() - 1),
                QPoint(x - 5, plot.top() - 8),
                QPoint(x + 5, plot.top() - 8),
            ]
        )

        painter.setPen(QPen(color, 1))
        label_x = max(10, min(self.width() - 60, x - 24))
        painter.drawText(label_x, self.height() - 24, text)

    def mousePressEvent(self, event):
        if not self.draggable:
            return

        x = int(event.position().x())
        bx = self._level_to_x(self.black_level)
        wx = self._level_to_x(self.white_level)

        self._drag_target = "black" if abs(x - bx) <= abs(x - wx) else "white"
        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event):
        if not self.draggable or self._drag_target is None:
            return

        level = self._x_to_level(int(event.position().x()))

        if self._drag_target == "black":
            self.set_levels(level, max(level + 1, self.white_level), emit_signal=True)
        elif self._drag_target == "white":
            self.set_levels(min(level - 1, self.black_level), level, emit_signal=True)

    def mouseReleaseEvent(self, event):
        self._drag_target = None
