# Document Restoration Pipeline

A modular, GPU-accelerated Python pipeline for restoring degraded scanned documents to OCR-ready quality.

## Features

- **8-stage pipeline**: Preprocessing → Illumination Correction → Stain Removal → Deblur → Super-Resolution → Text Restoration → Binarization → Post-Processing
- **YAML configuration**: Every parameter is tunable via `config.yaml`
- **GPU acceleration**: Automatic CUDA detection for Real-ESRGAN super-resolution
- **Batch processing**: Process entire directories of scanned documents
- **Timestamped logging**: Console + file logging with full diagnostics
- **Before/after comparisons**: Auto-generated side-by-side comparison images
- **Error handling**: Graceful per-file error handling for corrupt inputs

## Quick Start

### 1. Setup Environment

```powershell
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

Edit `config.yaml` to set input/output paths and tune parameters:

```yaml
paths:
  input_dir: "D:/PDF Restoration/Low_quality_samples"
  output_dir: "D:/PDF Restoration/Restored_output"
```

### 3. Run

```powershell
# Process all images in input directory
python restore_pipeline.py --config config.yaml

# Process a single image
python restore_pipeline.py --config config.yaml --input "path/to/image.png"
```

## Pipeline Stages

| Stage | Name | Method | Purpose |
|-------|------|--------|---------|
| 1 | Preprocessing | Bilateral filter, NLM denoise | Remove noise while preserving edges |
| 2 | Illumination Correction | Morphological background + CLAHE | Flatten uneven lighting |
| 3 | Stain & Bleed Removal | HSV segmentation + inpainting | Remove colour stains, suppress ghost text |
| 4 | Deblur | Wiener filter + unsharp mask | Sharpen blurred text |
| 5 | Super-Resolution | Real-ESRGAN 2× | Upscale low-resolution scans |
| 6 | Text Restoration | Morphological closing + CC analysis | Reconnect broken strokes, filter noise |
| 7 | Binarization | Sauvola adaptive threshold | Clean black text on white background |
| 8 | Post-Processing | Median filter, deskew, autocrop | Final cleanup and alignment |

## Configuration Reference

See `config.yaml` for all parameters. Key settings:

- `general.use_gpu`: Enable/disable GPU acceleration (default: `true`)
- `super_resolution.enabled`: Toggle Real-ESRGAN upscaling
- `binarization.method`: `sauvola` | `otsu` | `adaptive`
- `postprocessing.deskew.enabled`: Auto-straighten skewed scans

## Output

- Restored images: `Restored_output/<filename>_restored.png` (300 DPI)
- Comparisons: `Restored_output/comparisons/<filename>_comparison.png`
- Logs: `logs/restore_<timestamp>.log`

## Requirements

- Python 3.9+
- OpenCV 4.8+
- PyTorch 2.0+
- See `requirements.txt` for full list

### Optional

- **Tesseract OCR 5.x**: For OCR confidence verification (system-level install)
