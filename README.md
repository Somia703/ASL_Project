# ASL Project

Real-time American Sign Language character recognition using webcam input, MediaPipe hand detection, and a trained TensorFlow/Keras model.

## Features

- Detects one hand from webcam frames
- Crops a square hand ROI and runs model inference
- Supports 26-class and 36-class label mappings
- Optional live prediction smoothing with confidence threshold
- Snapshot prediction mode (`p`) with optional image saving in `captures/`

## Project Files

- `app.py`: main webcam inference app
- `asl_model.h5`: trained model file used for predictions
- `requirements.txt`: base dependencies

## Requirements

- Python 3.10+ (3.11 recommended)
- Webcam access

Install dependencies:

```bash
pip install -r requirements.txt
pip install opencv-python
```

## Run

```bash
python app.py
```

## Controls

- `c`: cycle preprocess mode (`BGR`, `RGB`, `GRAY3`, `GRAY_BIN3`)
- `n`: cycle normalization mode (`ZERO_ONE`, `NEG_ONE_ONE`)
- `p`: snapshot prediction of the latest hand crop
- `r`: reset smoothing buffer
- `Esc`: exit

## Useful Environment Variables

- `LIVE_PREDICT` (`0` or `1`): enable/disable continuous prediction overlay
- `MIN_CONFIDENCE` (default `0.70`): confidence gate for stable output
- `SMOOTHING_WINDOW` (default `8`): prediction smoothing window size
- `SHOW_TOPK` (default `3`): number of top classes shown in debug text
- `MISS_RESET_FRAMES` (default `6`): clears smoothing after consecutive misses
- `PREPROCESS_MODE`: initial preprocess mode
- `NORM_MODE`: initial normalization mode
- `SAVE_CAPTURES` (`0` or `1`): save snapshot crops to `captures/`
- `LABEL_ORDER`: set `DIGITS_FIRST` if your 36-class model uses `0-9` then `A-Z`

PowerShell example:

```powershell
$env:LIVE_PREDICT="1"
$env:MIN_CONFIDENCE="0.75"
python app.py
```

## Notes

- Ensure `asl_model.h5` is present in the project root before running.
- If MediaPipe landmarks are unavailable in your build, the app falls back to a fixed ROI box mode.
