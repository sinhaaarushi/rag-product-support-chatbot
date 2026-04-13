from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import config


def create_vector_store_backup() -> str:
    config.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = config.BACKUP_DIR / f"faiss_backup_{stamp}"
    archive = shutil.make_archive(str(out_file), "zip", root_dir=config.VECTOR_STORE_DIR)
    return archive


def restore_vector_store_backup(zip_path: str | Path) -> None:
    src = Path(zip_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Backup zip not found: {src}")
    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(src), extract_dir=str(config.VECTOR_STORE_DIR))
