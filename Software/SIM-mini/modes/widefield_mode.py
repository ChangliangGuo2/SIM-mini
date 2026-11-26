import cv2
import numpy as np
from PySide6.QtWidgets import (
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.constants import PROJECT_H, PROJECT_W


class WidefieldMode(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        group = QGroupBox("Widefield Illumination")
        group_layout = QVBoxLayout(group)

        self.full_white_btn = QPushButton("Widefield: Full White (640x480)")
        self.full_white_btn.clicked.connect(self.show_full_white)

        self.calibration_btn = QPushButton("Calibration: Up + Right Arrows")
        self.calibration_btn.clicked.connect(self.show_calibration_pattern)

        self.close_illumination_btn = QPushButton("Close Illumination")
        self.close_illumination_btn.clicked.connect(self.close_illumination)

        group_layout.addWidget(self.full_white_btn)
        group_layout.addWidget(self.calibration_btn)
        group_layout.addWidget(self.close_illumination_btn)

        layout.addWidget(group)
        layout.addStretch(1)

    def show_full_white(self):
        if not self.app.show_projection_white(warn=True):
            return
        self.app.set_status("Widefield full-white illumination shown on projection.")

    @staticmethod
    def create_calibration_pattern(width=PROJECT_W, height=PROJECT_H):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        thickness = max(12, min(width, height) // 18)

        cv2.arrowedLine(
            image,
            (int(width * 0.28), int(height * 0.78)),
            (int(width * 0.28), int(height * 0.16)),
            (255, 255, 255),
            thickness,
            cv2.LINE_8,
            tipLength=0.22,
        )
        cv2.arrowedLine(
            image,
            (int(width * 0.43), int(height * 0.67)),
            (int(width * 0.88), int(height * 0.67)),
            (255, 255, 255),
            thickness,
            cv2.LINE_8,
            tipLength=0.25,
        )
        return image

    def show_calibration_pattern(self):
        pattern = self.create_calibration_pattern()
        if not self.app.show_projection_pattern(pattern, warn=True):
            return
        self.app.set_status(
            "Calibration pattern shown: large up and right arrows."
        )

    def close_illumination(self):
        if not self.app.show_projection_black(warn=True):
            return
        self.app.set_status("Illumination closed: black image shown on projection.")
