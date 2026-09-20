import argparse
from pathlib import Path

from langchain_core.messages import HumanMessage

from config import DEFAULT_BATCH_SIZE
from agents.orchestrator.graph import build_graph
from logger import log_skip


def chunk(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def collect_source_files(src_root: Path) -> list[str]:
    files = []
    for p in src_root.rglob("*"):
        if p.is_file():
            files.append(str(p.relative_to(src_root)))
    return sorted(files)


def make_batch_prompt(batch_files: list[str]) -> str:
    listing = "\n".join(f"- {f}" for f in batch_files)
    return f"Here are the files in this batch:\n{listing}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Local file organizer agent")
    parser.add_argument("src", type=Path, help="Source directory")
    parser.add_argument("dst", type=Path, help="Destination directory")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    src_root = args.src.resolve()
    dst_root = args.dst.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    # Pre-chunked once at startup (per your call) — simpler, at the cost of
    # not picking up files an earlier batch left behind mid-run. Files are
    # never re-scanned or re-queued.
    all_files = collect_source_files(src_root)
    if not all_files:
        print("No files found in source directory.")
        return

    graph = build_graph()

    for batch_num, batch_files in enumerate(chunk(all_files, args.batch_size), start=1):
        print(f"\n--- Batch {batch_num} ({len(batch_files)} files) ---")

        moved_files: list[str] = []  # mutated in place by move_file calls
        initial_state = {"messages": [HumanMessage(make_batch_prompt(batch_files))]}
        run_config = {
            "configurable": {
                "src_root": src_root,
                "dst_root": dst_root,
                "moved_files": moved_files,
            },
            "recursion_limit": 50,  # safety net against a runaway tool loop
        }

        # Fresh invoke, fresh message history — no checkpointer, so this
        # is functionally "a new agent instance" each batch, as intended.
        graph.invoke(initial_state, config=run_config)

        # Anything not explicitly moved is logged as skipped here,
        # deterministically — never trusting the LLM's self-report.
        skipped = set(batch_files) - set(moved_files)
        for rel_path in sorted(skipped):
            log_skip(rel_path)

        print(f"Moved: {len(moved_files)}, Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
