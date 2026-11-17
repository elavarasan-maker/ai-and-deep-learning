from pathlib import Path
import zipfile
import requests
from tqdm import tqdm

URL = "https://zenodo.org/records/10881058/files/Sen2Fire.zip?download=1"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARCHIVE = DATA / "Sen2Fire.zip"
DATA.mkdir(parents=True, exist_ok=True)

if not ARCHIVE.exists():
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with ARCHIVE.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading Sen2Fire") as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
else:
    print(f"Archive already exists: {ARCHIVE}")

print("Extracting archive (about 6.3 GB compressed)...")
with zipfile.ZipFile(ARCHIVE) as z:
    z.extractall(DATA)
print(f"Extraction complete under: {DATA}")
