import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core import camera
from modes.sim_mode import PatternGenerator, SIMMode


class FakeCapture:
    def __init__(self, *_args):
        self.values = {
            camera.cv2.CAP_PROP_AUTO_EXPOSURE: 0.75,
            camera.cv2.CAP_PROP_EXPOSURE: -5.0,
            camera.cv2.CAP_PROP_GAIN: 10.0,
        }

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.values[prop] = float(value)
        return True

    def get(self, prop):
        return self.values[prop]

    def release(self):
        pass


class FakeCaptureWithoutAutoExposureReadback(FakeCapture):
    def set(self, prop, value):
        if prop != camera.cv2.CAP_PROP_AUTO_EXPOSURE:
            self.values[prop] = float(value)
        return True

    def get(self, prop):
        if prop == camera.cv2.CAP_PROP_AUTO_EXPOSURE:
            return -1.0
        return super().get(prop)


class CameraControlTests(unittest.TestCase):
    def test_windows_controls_are_set_and_verified(self):
        with patch.object(camera.cv2, "VideoCapture", FakeCapture):
            report = camera._apply_windows_fixed_camera_controls(
                "test camera", 0, -1.0, 12.0
            )

        self.assertTrue(report["verified"])
        self.assertEqual(report["readback"]["auto_exposure"], 0.25)
        self.assertEqual(report["readback"]["exposure"], -1.0)
        self.assertEqual(report["readback"]["gain"], 12.0)

    def test_windows_driver_can_accept_manual_mode_without_readback(self):
        with patch.object(
            camera.cv2,
            "VideoCapture",
            FakeCaptureWithoutAutoExposureReadback,
        ):
            report = camera._apply_windows_fixed_camera_controls(
                "test camera", 0, 0.0, 0.0
            )

        self.assertTrue(report["verified"])
        self.assertEqual(
            report["auto_exposure_verification"],
            "set_accepted_readback_unsupported",
        )
        self.assertTrue(report["warnings"])

    def test_linux_controls_are_set_and_read_back(self):
        control_values = {
            "exposure_auto": 1,
            "exposure_absolute": 120,
            "gain": 8,
        }

        def fake_v4l2(_device, *args):
            if args == ("--list-ctrls",):
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "exposure_auto 0x009a0901 (menu)\n"
                        "exposure_absolute 0x009a0902 (int)\n"
                        "gain 0x00980913 (int)\n"
                    ),
                    stderr="",
                )
            if args[0] == "--set-ctrl":
                name, raw_value = args[1].split("=", 1)
                control_values[name] = int(raw_value)
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            name = args[1]
            return SimpleNamespace(
                returncode=0,
                stdout=f"{name}: {control_values[name]}\n",
                stderr="",
            )

        with patch.object(camera, "_run_v4l2_control", side_effect=fake_v4l2):
            report = camera._apply_linux_fixed_camera_controls(
                "/dev/video0", 120, 8
            )

        self.assertTrue(report["verified"])
        self.assertEqual(report["readback"]["exposure"], 120.0)
        self.assertEqual(report["readback"]["gain"], 8.0)


class SimAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.app = SimpleNamespace(
            current_camera_stream_format=lambda: "yuyv422",
            camera_acquisition_metadata=lambda: {
                "device_name": "test camera",
                "stream_format": "yuyv422",
                "fixed_exposure_gain": {"verified": True},
            },
        )
        self.mode = SIMMode(self.app)

    def test_conventional_sim_discards_four_extra_first_pattern_frames(self):
        self.mode.index = 0
        self.assertEqual(self.mode._pattern_discard_frames(), 6)
        self.mode.index = 1
        self.assertEqual(self.mode._pattern_discard_frames(), 2)

    def test_metadata_contains_protocol_camera_and_frame_records(self):
        self.mode.patterns = PatternGenerator.generate_sim_sequence(
            stripe_width=3,
            width=640,
            height=480,
        )
        self.mode.acquisition_created_at = "2026-09-08 12:00:00"
        self.mode.source_dtype = "uint8"
        self.mode.raw_widefield_record = {"captured_sequence": 10}
        self.mode.captured_pattern_records = [
            {"frame_index": 1, "captured_sequence": 17, "discard_frames": 6}
        ]
        self.mode.index = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            self.mode.output_dir = Path(temp_dir)
            self.mode._write_acquisition_metadata(status="running")
            metadata = json.loads(
                (Path(temp_dir) / "acquisition_metadata.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(metadata["metadata_version"], 2)
        self.assertTrue(metadata["camera"]["fixed_exposure_gain"]["verified"])
        self.assertEqual(
            metadata["acquisition_parameters"]["first_pattern_extra_discard_frames"],
            4,
        )
        self.assertEqual(len(metadata["pattern_sequence"]), 9)
        self.assertEqual(metadata["pattern_frame_captures"][0]["captured_sequence"], 17)


if __name__ == "__main__":
    unittest.main()
