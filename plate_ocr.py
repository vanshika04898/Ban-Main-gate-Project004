import os
import re
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# ---- config ----
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "license_plate_detector.pt")
OCR_LANGS = os.getenv("OCR_LANGS", "en").split(",")
MIN_OCR_CONF = float(os.getenv("MIN_OCR_CONF", "0.4"))
PLATE_REGEX = re.compile(os.getenv("PLATE_REGEX", r"[A-Z0-9]{4,12}"))

# ---- load models once (CPU) ----
_plate_detector = YOLO(YOLO_MODEL_PATH)
_ocr_reader = easyocr.Reader(OCR_LANGS, gpu=False)


def _decode_base64_image(b64: str) -> np.ndarray:
    """Return BGR image (OpenCV). Raises ValueError on invalid input."""
    if not b64 or not isinstance(b64, str):
        raise ValueError("image_base64 must be a non-empty string")

    # allow data URL prefix
    if b64.strip().lower().startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(b64, validate=True)
    except Exception:
        img_bytes = base64.b64decode(b64 + "===")

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid base64 image (decode failed)")
    return img


def _preprocess_plate(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )
    return thr


def _normalize_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def _best_plate_box(image_bgr: np.ndarray):
    """Return (x1,y1,x2,y2,conf) for best plate box or None."""
    res = _plate_detector.predict(image_bgr, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None

    confs = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy().astype(int)

    i = int(np.argmax(confs))
    x1, y1, x2, y2 = xyxy[i]
    return x1, y1, x2, y2, float(confs[i])


def extract_plate_from_base64(image_base64: str) -> dict:
    """
    Returns:
      {
        "plate": str|None,
        "plate_confidence": float|None,
        "detection_confidence": float|None,
        "bbox": {x1,y1,x2,y2}|None
      }
    """
    image = _decode_base64_image(image_base64)

    best = _best_plate_box(image)
    if not best:
        return {"plate": None, "plate_confidence": None, "detection_confidence": None, "bbox": None}

    x1, y1, x2, y2, det_conf = best
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return {"plate": None, "plate_confidence": None, "detection_confidence": det_conf, "bbox": None}

    proc = _preprocess_plate(crop)

    ocr = _ocr_reader.readtext(proc)
    candidates = []
    for _, text, conf in ocr:
        clean = _normalize_text(text)
        if conf >= MIN_OCR_CONF and PLATE_REGEX.fullmatch(clean):
            candidates.append((clean, float(conf)))

    if not candidates:
        return {
            "plate": None,
            "plate_confidence": None,
            "detection_confidence": det_conf,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }

    plate, ocr_conf = max(candidates, key=lambda x: x[1])
    return {
        "plate": plate,
        "plate_confidence": ocr_conf,
        "detection_confidence": det_conf,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }