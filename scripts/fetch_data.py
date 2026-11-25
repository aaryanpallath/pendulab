"""Download the real double-pendulum recordings and save the angles as data/pendulum.npz.

Source: A. Myers, F. Khasawneh, J. Tempelman, D. Petrushenko, "Low-cost double pendulum for
high-quality data collection with open-source video tracking and analysis", Mendeley Data,
V1 (2020), doi:10.17632/7yd2ntbh3w.1. License: CC BY 4.0.

Only the video-tracked mean angle of each link is kept (columns: time s, angle deg),
converted to radians. Nothing in the archive is executed.
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = ("https://data.mendeley.com/public-files/datasets/7yd2ntbh3w/files/"
       "91cc2fa5-2640-404b-8696-05f0aede2f88/file_downloaded")
SHA256 = "1e998d549726528d36a0de81f7e93b8756b10b53d18b2bbeaf08194fb22b8564"
OUT = Path(__file__).resolve().parent.parent / "data" / "pendulum.npz"


def convert(raw: bytes) -> dict[str, np.ndarray]:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch: got {digest}, expected {SHA256}")
    out = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for trial in (1, 2, 3):
            cols = []
            for link in (0, 1):
                name = f"Video_Tracking_Data/Trial{trial}/DPmean_data_RB{link}.csv"
                arr = np.loadtxt(io.TextIOWrapper(z.open(name)), delimiter=",")
                cols.append(arr)
            t = cols[0][:, 0]
            if not np.array_equal(t, cols[1][:, 0]):
                raise SystemExit(f"trial {trial}: link time stamps differ")
            out[f"trial{trial}_t"] = t
            out[f"trial{trial}_q"] = np.deg2rad(np.stack([cols[0][:, 1], cols[1][:, 1]], axis=1))
    return out


def main(argv: list[str]) -> int:
    src = argv[1] if len(argv) > 1 else None
    raw = Path(src).read_bytes() if src else urllib.request.urlopen(URL, timeout=120).read()
    data = convert(raw)
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **data)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
