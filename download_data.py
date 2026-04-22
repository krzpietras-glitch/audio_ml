"""
Download and extract the ESC-50 dataset (~600 MB).
Run from the project root:
    python download_data.py
"""

import os
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path("data")
URL      = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
ZIP_PATH = DATA_DIR / "ESC-50-master.zip"
EXPECTED = DATA_DIR / "ESC-50-master"


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ESC-50 → {dest}")
    print("(~600 MB — this may take a few minutes on a slow connection)")

    def _progress(count, block_size, total_size):
        pct = count * block_size / total_size * 100
        mb  = count * block_size / 1_000_000
        print(f"\r  {pct:.1f}%  {mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print()


def extract(zip_path: Path, dest_dir: Path):
    print(f"Extracting {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print("Done.")


def verify(path: Path):
    audio_dir = path / "audio"
    meta_csv  = path / "meta" / "esc50.csv"
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing: {audio_dir}")
    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing: {meta_csv}")
    n_wavs = len(list(audio_dir.glob("*.wav")))
    print(f"ESC-50 verified: {n_wavs} WAV files found at {audio_dir}")


def main():
    if EXPECTED.exists():
        print(f"ESC-50 already present at {EXPECTED}")
        verify(EXPECTED)
        return

    download(URL, ZIP_PATH)
    extract(ZIP_PATH, DATA_DIR)

    # Remove zip to save space (optional — comment out to keep)
    os.remove(ZIP_PATH)
    print(f"Removed {ZIP_PATH}")

    verify(EXPECTED)


if __name__ == "__main__":
    main()
