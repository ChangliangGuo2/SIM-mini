from pathlib import Path
import json
import csv

import numpy as np
import tifffile as tiff
from PySide6.QtCore import QThread, Signal

from .image_io import imwrite_unicode


class SaveWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, tasks, label="Save", parent=None):
        super().__init__(parent)
        self.tasks = list(tasks)
        self.label = str(label)

    def run(self):
        try:
            total = max(1, len(self.tasks))

            for i, task in enumerate(self.tasks, start=1):
                kind = task.get("kind", "image")
                path = Path(task["path"])
                path.parent.mkdir(parents=True, exist_ok=True)

                if kind == "image":
                    ok = imwrite_unicode(path, task["data"])
                    if not ok:
                        raise RuntimeError(f"Failed to save image: {path}")

                elif kind == "bytes":
                    data = task.get("data", b"")
                    if isinstance(data, str):
                        data = data.encode("utf-8")
                    path.write_bytes(data)

                elif kind == "tiff":
                    kwargs = dict(task.get("kwargs", {}))
                    data = task["data"]
                    tiff.imwrite(str(path), data, **kwargs)

                elif kind == "csv_rows":
                    header = task.get("header", [])
                    rows = task.get("rows", [])
                    with open(path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if header:
                            writer.writerow(header)
                        writer.writerows(rows)

                elif kind == "json":
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(task.get("data", {}), f, ensure_ascii=False, indent=2)

                elif kind == "npy":
                    np.save(path, task["data"])

                else:
                    raise ValueError(f"Unknown save task kind: {kind}")

                pct = int(i / total * 100)
                self.progress.emit(pct, f"{self.label}: saved {path.name}")

            self.finished.emit(self.label)

        except Exception as e:
            self.error.emit(str(e))
