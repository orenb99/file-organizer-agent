import argparse
import subprocess
from pathlib import Path

from langchain_core.messages import HumanMessage

from config import DEFAULT_BATCH_SIZE, DEFAULT_SOURCE_DIR, DEFAULT_DEST_DIR
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


def prompt_for_instruction() -> str:
    # Only asked once, at boot, when --instruction wasn't passed on the
    # CLI — so scripted/cron runs (which pass --instruction "" or omit
    # stdin) don't hang waiting for input.
    try:
        return input(
            "Any additional instructions for the agent this run? (Enter to skip): "
        ).strip()
    except EOFError:
        return ""


def run_batch(graph, initial_state: dict, run_config: dict, verbose: bool) -> None:
    """Run one batch through the graph. Verbose mode streams state after
    every node and pretty-prints any messages not yet shown — this covers
    the agent's replies, its tool calls (args included), and the tool
    results, in order. Non-verbose mode just invokes silently."""
    if not verbose:
        graph.invoke(initial_state, config=run_config)
        return

    printed = 0
    for state in graph.stream(initial_state, config=run_config, stream_mode="values"):
        messages = state["messages"]
        for msg in messages[printed:]:
            msg.pretty_print()
        printed = len(messages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local file organizer agent")
    parser.add_argument("src", type=Path, nargs="?", default=DEFAULT_SOURCE_DIR, help="Source directory")
    parser.add_argument("dst", type=Path, nargs="?", default=DEFAULT_DEST_DIR, help="Destination directory")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print agent messages and tool calls as they happen",
    )
    parser.add_argument(
        "-i", "--instruction", type=str, default=None,
        help="Extra instruction for the agent this run (skips the interactive prompt)",
    )
    args = parser.parse_args()

    src_root = args.src.resolve()
    dst_root = args.dst.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    custom_instruction = args.instruction if args.instruction is not None else prompt_for_instruction()

    # Pre-chunked once at startup (per your call) — simpler, at the cost of
    # not picking up files an earlier batch left behind mid-run. Files are
    # never re-scanned or re-queued.
    all_files = collect_source_files(src_root)
    if not all_files:
        print("No files found in source directory.")
        return

    graph = build_graph(verbose=args.verbose)
    subprocess.Popen(rf'explorer /select,"{dst_root}/."')
    subprocess.Popen(rf'explorer /select,"{src_root}/."')

    for batch_num, batch_files in enumerate(chunk(all_files, args.batch_size), start=1):
        print(f"\n--- Batch {batch_num} ({len(batch_files)} files) ---")

        moved_files: list[str] = []  # mutated in place by move_file calls
        initial_state = {"messages": [HumanMessage(make_batch_prompt(batch_files))]}
        run_config = {
            "configurable": {
                "src_root": src_root,
                "dst_root": dst_root,
                "moved_files": moved_files,
                "custom_instruction": custom_instruction,
            },
            "recursion_limit": 50,  # safety net against a runaway tool loop
        }

        # Fresh invoke, fresh message history — no checkpointer, so this
        # is functionally "a new agent instance" each batch, as intended.
        run_batch(graph, initial_state, run_config, args.verbose)

        # Anything not explicitly moved is logged as skipped here,
        # deterministically — never trusting the LLM's self-report.
        skipped = set(batch_files) - set(moved_files)
        for rel_path in sorted(skipped):
            log_skip(rel_path)

        print(f"Moved: {len(moved_files)}, Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()