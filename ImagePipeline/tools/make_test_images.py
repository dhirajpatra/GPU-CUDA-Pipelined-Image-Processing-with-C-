#!/usr/bin/env python3
"""Generate a BMP test set with mixed resolutions: python make_test_images.py <out_dir> [count]"""
import struct, sys
from pathlib import Path
import numpy as np

SIZES = [(640, 480), (1280, 720), (800, 600), (1920, 1080), (3840, 2160), (1024, 1024), (333, 217)]

def write_bmp(path, rgb):
    h, w, _ = rgb.shape
    bgr = np.ascontiguousarray(rgb[::-1, :, ::-1])            # bottom-up, BGR
    row = (w * 3 + 3) // 4 * 4
    padded = np.zeros((h, row), np.uint8); padded[:, : w * 3] = bgr.reshape(h, w * 3)
    size = 54 + row * h
    hdr = b"BM" + struct.pack("<IHHIIiiHHIIiiII", size, 0, 0, 54, 40, w, h, 1, 24, 0, row * h, 2835, 2835, 0, 0)
    path.write_bytes(hdr + padded.tobytes())

def make(w, h, rng):
    img = np.full((h, w, 3), rng.integers(10, 40), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(rng.integers(4, 12)):
        v = rng.integers(150, 250)
        if rng.random() < 0.5:                               # rectangle
            rw, rh = rng.integers(w // 14, w // 5), rng.integers(h // 14, h // 5)
            x0, y0 = rng.integers(8, w - rw - 8), rng.integers(8, h - rh - 8)
            img[y0:y0 + rh, x0:x0 + rw] = v
        else:                                                # disc
            r = rng.integers(min(w, h) // 20, min(w, h) // 8)
            cx, cy = rng.integers(r + 8, w - r - 8), rng.integers(r + 8, h - r - 8)
            img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = v
    noise = rng.normal(0, 2.0, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    rng = np.random.default_rng(42)
    for i in range(count):
        w, h = SIZES[i % len(SIZES)]
        write_bmp(out / f"img_{i:03d}_{w}x{h}.bmp", make(w, h, rng))
    print(f"wrote {count} images to {out}")
