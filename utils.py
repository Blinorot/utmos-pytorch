from pathlib import Path

TARGET_SR = 16000

ROOT_DIR = Path(__file__).resolve().parent
ASSET_DATA_DIR = ROOT_DIR / "data" / "assets"
SCRIPT_DATA_DIR = ROOT_DIR / "data" / "script"
ASSET_DATA_DIR.mkdir(exist_ok=True, parents=True)
SCRIPT_DATA_DIR.mkdir(exist_ok=True, parents=True)
