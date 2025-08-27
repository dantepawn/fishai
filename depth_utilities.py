import numpy as np
import cv2

def _binarize(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    return m

def _crop_union_roi(a: np.ndarray, b: np.ndarray, pad: int = 20):
    def bbox(m):
        pts = cv2.findNonZero(m)
        if pts is None:
            return None
        x, y, w, h = cv2.boundingRect(pts)
        return x, y, x+w, y+h

    ha, wa = a.shape[:2]
    bb_a = bbox(a)
    bb_b = bbox(b)
    if bb_a is None or bb_b is None:
        return (a, b, (0, 0))  # nothing to crop; will handle as-is

    x1a, y1a, x2a, y2a = bb_a
    x1b, y1b, x2b, y2b = bb_b
    x1 = max(0, min(x1a, x1b) - pad)
    y1 = max(0, min(y1a, y1b) - pad)
    x2 = min(wa, max(x2a, x2b) + pad)
    y2 = min(ha, max(y2a, y2b) + pad)
    return a[y1:y2, x1:x2], b[y1:y2, x1:x2], (x1, y1)

def _signed_distance(mask_bin: np.ndarray) -> np.ndarray:
    # cv2.distanceTransform expects 8-bit, non-zero as foreground
    fg = mask_bin.astype(np.uint8)
    bg = (1 - mask_bin).astype(np.uint8)
    dt_f = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    dt_b = cv2.distanceTransform(bg, cv2.DIST_L2, 3)
    sdt = dt_f - dt_b
    # normalize to [-1, 1] for stability
    maxv = np.max(np.abs(sdt)) + 1e-6
    return (sdt / maxv).astype(np.float32)

def _hann2d(h, w):
    wx = np.hanning(w)
    wy = np.hanning(h)
    return (wy[:, None] * wx[None, :]).astype(np.float32)

def _phase_corr(a: np.ndarray, b: np.ndarray):
    # returns (dx, dy, response)
    h, w = a.shape
    win = _hann2d(h, w)
    shift, response = cv2.phaseCorrelate(a * win, b * win)
    dx, dy = shift  # OpenCV returns (dx, dy)
    return float(dx), float(dy), float(response)

def _shift_mask_binary(mask: np.ndarray, dy: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)
    if dy > 0:
        out[dy:, :] = mask[:h-dy, :]
    elif dy < 0:
        out[:h+dy, :] = mask[-dy:, :]
    else:
        out[:] = mask
    return out

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-6)

def estimate_mask_shift(mask_a: np.ndarray,
                        mask_b: np.ndarray,
                        refine_window: int = 6,
                        ecc_iters: int = 1000,
                        ecc_eps: float = 1e-6):
    """
    Returns: dy, dx, score, method
    dy > 0 means mask_a is below mask_b (shift a down aligns to b).
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError("Masks must have the same shape")

    # Binarize and crop to ROI
    a_bin = _binarize(mask_a)
    b_bin = _binarize(mask_b)
    a_roi, b_roi, (x0, y0) = _crop_union_roi(a_bin, b_bin, pad=20)

    # Handle empty masks
    if a_roi.sum() == 0 or b_roi.sum() == 0:
        return None, None, 0.0, "invalid"

    # Build SDTs for robustness to shape differences
    a_sdt = _signed_distance(a_roi)
    b_sdt = _signed_distance(b_roi)

    # Try ECC (translation model)
    method = "ecc_sdt"
    dy = dx = 0.0
    score = 0.0
    try:
        warp_mode = cv2.MOTION_TRANSLATION
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iters, ecc_eps)
        cc, warp = cv2.findTransformECC(templateImage=b_sdt, inputImage=a_sdt,
                                        warpMatrix=warp, motionType=warp_mode,
                                        criteria=criteria, inputMask=None, gaussFiltSize=5)
        dx = float(warp[0, 2])
        dy = float(warp[1, 2])
        score = float(cc)
    except cv2.error:
        # Fallback to phase correlation
        method = "phasecorr_sdt"
        dx, dy, resp = _phase_corr(a_sdt, b_sdt)
        score = resp

    # Optional small-window IoU refine on vertical shift only
    # Round to nearest int and sweep ±refine_window
    base_dy = int(round(dy))
    best_iou = -1.0
    best_dy = base_dy
    for ddy in range(base_dy - refine_window, base_dy + refine_window + 1):
        a_shift = _shift_mask_binary(a_roi, ddy)
        iou = _iou(a_shift.astype(bool), b_roi.astype(bool))
        if iou > best_iou:
            best_iou = iou
            best_dy = ddy

    # Replace dy with refined value; keep dx from ECC/phase if needed
    dy_refined = float(best_dy)
    return dy_refined, dx, max(score, best_iou), method