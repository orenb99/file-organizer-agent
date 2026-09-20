from datetime import datetime

from config import LOG_FILE


def _write(line: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_move(src_rel: str, dest_rel: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    _write(f"{ts} MOVE {src_rel} -> {dest_rel}")


def log_skip(rel_path: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    _write(f"{ts} SKIP {rel_path}")
