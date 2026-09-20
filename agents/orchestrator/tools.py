import shutil
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from logger import log_move
from agents.orchestrator.path_utils import resolve_and_validate


def _cfg(config: RunnableConfig, key: str):
    return config["configurable"][key]


@tool
def make_dir(path: str, config: RunnableConfig) -> str:
    """Create a directory inside the destination root. `path` is relative
    to the destination root, e.g. 'Documents/Invoices'. Parent directories
    are created automatically if needed."""
    dst_root: Path = _cfg(config, "dst_root")
    target = resolve_and_validate(dst_root, path)
    target.mkdir(parents=True, exist_ok=True)
    return f"Created directory: {path}"


@tool
def nested_ls(path: str, config: RunnableConfig, depth: int = 3) -> str:
    """List the file/directory tree under `path` (relative to the
    destination root) up to `depth` levels deep. Use this to see what's
    already organized in the destination before inventing a new folder,
    so you stay consistent with structure from earlier batches."""
    dst_root: Path = _cfg(config, "dst_root")
    base = resolve_and_validate(dst_root, path)
    if not base.exists():
        return f"'{path}' does not exist yet in the destination."

    lines: list[str] = []

    def walk(dir_path: Path, level: int) -> None:
        if level > depth:
            return
        for entry in sorted(dir_path.iterdir()):
            rel = entry.relative_to(dst_root)
            marker = "/" if entry.is_dir() else ""
            lines.append(f"{'  ' * level}{rel.name}{marker}")
            if entry.is_dir():
                walk(entry, level + 1)

    walk(base, 0)
    return "\n".join(lines) if lines else "(empty)"


@tool
def move_file(src_rel: str, dest_rel: str, config: RunnableConfig) -> str:
    """Move a file from the source root to the destination root. `src_rel`
    must be one of the files given to you in this batch. `dest_rel` is
    where it should end up, relative to the destination root — parent
    directories are created automatically if they don't already exist."""
    src_root: Path = _cfg(config, "src_root")
    dst_root: Path = _cfg(config, "dst_root")

    src_path = resolve_and_validate(src_root, src_rel)
    dest_path = resolve_and_validate(dst_root, dest_rel)

    if not src_path.is_file():
        raise ValueError(f"'{src_rel}' is not a file in the source root.")

    # Auto-create parent dirs on move (per your call) — the agent doesn't
    # need a separate make_dir round-trip for every single leaf file.
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(src_path), str(dest_path))

    # Logging happens here, deterministically, not via the LLM narrating
    # what it did — the log file is only as trustworthy as the code
    # writing it, never the agent's self-report.
    log_move(src_rel, dest_rel)

    # Recorded in a list shared across this whole batch (injected via
    # config) so main.py can diff batch_files against it afterward and
    # find anything the agent left un-moved, without the agent needing
    # to explicitly declare a "skip".
    _cfg(config, "moved_files").append(src_rel)

    return f"Moved '{src_rel}' -> '{dest_rel}'"


TOOLS = [make_dir, nested_ls, move_file]
