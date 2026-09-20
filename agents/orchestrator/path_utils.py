from pathlib import Path


def resolve_and_validate(root: Path, rel_path: str) -> Path:
    """Resolve `rel_path` against `root`, refusing anything that escapes it
    (e.g. via '..'). The agent only ever sees relative paths; this is the
    single choke point that turns them into real, safe absolute paths.

    Raises ValueError on violation — that message becomes the ToolMessage
    content the agent sees, so it's written to be actionable, not just
    descriptive of the failure.
    """
    root = root.resolve()
    candidate = (root / rel_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(
            f"'{rel_path}' resolves outside the allowed root. "
            "Paths must be relative to the root you were given, with no '..'."
        )
    return candidate
