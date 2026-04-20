"""Zip-based backup and restore for the local FAISS store.

Keeping backups in a dated archive (instead of a live copy of the directory)
means a rebuild can't half-overwrite a backup if it crashes partway through,
and the user can keep several dated snapshots around without managing paths.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import config


def create_vector_store_backup() -> str:
    """Zip ``Data/vector_store/`` into a timestamped archive under ``Data/backups/``.

    Returns the absolute path to the created archive. ``shutil.make_archive``
    appends the format suffix itself so the base name passed in has no ``.zip``.
    """
    config.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_base = config.BACKUP_DIR / f"faiss_backup_{timestamp}"
    return shutil.make_archive(str(archive_base), "zip", root_dir=config.VECTOR_STORE_DIR)


def restore_vector_store_backup(zip_path: str | Path) -> None:
    """Unpack a backup archive on top of the current vector store.

    Does not clear the existing store first; callers that want a clean slate
    should delete the store contents before restoring, otherwise the restored
    archive will merge with whatever's already there.
    """
    archive_path = Path(zip_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Backup zip not found: {archive_path}")
    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive_path), extract_dir=str(config.VECTOR_STORE_DIR))
