import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from restore_pipeline import DocumentRestorer


def test_restore_single_writes_markdown_when_ocr_fails(tmp_path):
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "output"
    config_data = {
        "paths": {
            "input_dir": str(tmp_path),
            "output_dir": str(output_dir),
            "log_dir": str(tmp_path / "logs"),
            "comparison_dir": str(output_dir / "comparisons"),
        },
        "general": {
            "output_format": "png",
            "output_dpi": 300,
            "use_gpu": False,
            "log_level": "INFO",
        },
        "ocr": {
            "enabled": True,
            "engine": "chandra",
            "method": "hf",
            "model_checkpoint": "datalab-to/chandra-ocr-2",
            "save_txt": True,
            "save_format": "markdown",
            "use_mock_fallback": False,
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    image_path = tmp_path / "sample.png"
    cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))

    restorer = DocumentRestorer(str(config_path))
    restorer.ocr_engine.enabled = True
    restorer.ocr_engine.manager = None
    restorer.ocr_engine.use_mock_fallback = False
    restorer.ocr_engine.save_txt = True
    restorer.ocr_engine.save_format = "markdown"

    with patch.object(restorer, "restore_image_data", return_value=(np.zeros((10, 10, 3), dtype=np.uint8), None)):
        restorer.restore_single(str(image_path), base_input_dir=str(tmp_path))

    markdown_path = output_dir / "sample.md"
    assert markdown_path.exists(), f"Expected markdown output at {markdown_path}"
    content = markdown_path.read_text(encoding="utf-8")
    assert "OCR" in content or "Transcription" in content
