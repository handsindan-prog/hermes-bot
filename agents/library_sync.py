"""
library_sync — ingest a document library into a Hermes workspace.

Walks a folder of real files, extracts text, and stores each document under a
stable key. Re-running is free for anything that has not changed: store_document
hashes the source text and skips unchanged documents without embedding them
again, so this is safe to run on a timer.

The source is behind a seam. Today it is a mounted Google Drive folder read
from disk; the same pipeline takes a DriveApiSource later, when a service
account is a member of a Shared Drive and the sync no longer depends on anyone's
laptop being awake. Everything after fetch — extraction, chunking, embedding,
dedupe, workspace tagging, run logging — is source-agnostic and does not change.

    python agents/library_sync.py --root /tmp/lib --workspace vantage --dry-run
    python agents/library_sync.py --root /tmp/lib --workspace vantage --label "Priimal"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_core as hc

AGENT = "library-sync"

# Everything hermes_core can read. Anything else is reported as skipped rather
# than ignored silently — an unreadable file you don't know about is a gap in
# the corpus that looks like an absence of evidence.
READABLE = set(hc.EXTRACTORS)

# Rough bytes per stored row: ~1.5KB of text plus a 1024-dim vector at 4 bytes
# a dimension, plus index overhead. Used only to warn before filling a 500MB
# free tier, so approximate is fine.
BYTES_PER_ROW = 6_000
FREE_TIER_BYTES = 500 * 1024 * 1024


@dataclass
class Doc:
    """One source document, independent of where it came from."""
    key: str                 # stable identity within the workspace
    title: str
    path: Path
    ext: str
    size: int
    modified: str

    def text(self) -> str:
        return hc.EXTRACTORS[self.ext](str(self.path))


@dataclass
class FolderSource:
    """A mounted folder — Google Drive for Desktop, or anything else on disk."""
    root: Path
    label: str = ""
    skip_dirs: tuple = (".git", "__pycache__", ".Trash")

    skipped: list = field(default_factory=list)

    def docs(self) -> list[Doc]:
        out: list[Doc] = []
        for p in sorted(self.root.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            if any(part in self.skip_dirs for part in p.parts):
                continue

            ext = p.suffix.lower()
            rel = p.relative_to(self.root)

            if ext not in READABLE:
                self.skipped.append((str(rel), ext or "no extension"))
                continue

            # Key on the path rather than the filename: two folders can hold a
            # "Brief.pdf" and they are different documents. Prefixed so a row's
            # origin is legible in the archive without a lookup.
            prefix = f"{self.label}/" if self.label else ""
            out.append(Doc(
                key=f"drive:{prefix}{rel}",
                title=p.stem,
                path=p,
                ext=ext,
                size=p.stat().st_size,
                modified=datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc).isoformat()[:19],
            ))
        return out


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


async def survey(docs: list[Doc]) -> tuple[int, int, list[tuple]]:
    """Extract and chunk without embedding or storing, so the cost of a real
    run can be stated before anyone commits to it."""
    rows = 0
    words = 0
    detail = []
    for d in docs:
        try:
            text = d.text()
        except Exception as e:
            detail.append((d, 0, f"unreadable: {type(e).__name__}"))
            continue
        chunks = hc.chunk_text(text)
        w = len(text.split())
        rows += len(chunks)
        words += w
        detail.append((d, len(chunks), "" if chunks else "no extractable text"))
    return rows, words, detail


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder to ingest")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--label", default="", help="prefix for doc keys, e.g. 'Priimal'")
    ap.add_argument("--dry-run", action="store_true",
                    help="survey only: what would be ingested and what it costs")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.root).expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    src = FolderSource(root, label=a.label)
    docs = src.docs()
    if a.limit:
        docs = docs[:a.limit]

    if not docs:
        print(f"Nothing readable under {root}")
        if src.skipped:
            print(f"({len(src.skipped)} files skipped)")
        return 0

    # ---- survey, always. A run whose size nobody checked is how a free tier
    # ---- gets filled by accident.
    rows, words, detail = await survey(docs)

    existing = len(hc.supabase.table("memories").select("id").execute().data)
    projected = existing + rows

    print(f"Library survey — {root}")
    print(f"  workspace       {a.workspace}")
    print(f"  readable files  {len(docs)}")
    print(f"  skipped         {len(src.skipped)}")
    print(f"  words           {words:,}")
    print(f"  chunks to store {rows:,}")
    print()
    print(f"  corpus now      {existing:,} rows  (~{human(existing * BYTES_PER_ROW)})")
    print(f"  after this sync {projected:,} rows  (~{human(projected * BYTES_PER_ROW)})")
    print(f"  free tier       {human(FREE_TIER_BYTES)}  "
          f"→ {projected * BYTES_PER_ROW / FREE_TIER_BYTES * 100:.1f}% used")
    print()

    empties = [(d, why) for d, n, why in detail if why]
    if empties:
        print("  Files that yielded nothing:")
        for d, why in empties:
            print(f"    - {d.key.removeprefix('drive:')}  ({why})")
        print()

    if src.skipped:
        by_ext: dict[str, int] = {}
        for _, ext in src.skipped:
            by_ext[ext] = by_ext.get(ext, 0) + 1
        print("  Skipped, by type: " +
              ", ".join(f"{k} ×{v}" for k, v in sorted(by_ext.items(), key=lambda x: -x[1])))
        print()

    if a.dry_run:
        print("  Largest documents:")
        for d, n, _ in sorted(detail, key=lambda x: -x[1])[:10]:
            print(f"    {n:>4} chunks  {d.key.removeprefix('drive:')}")
        print("\nDry run — nothing stored.")
        return 0

    # ---- real run ----
    counts = {"stored": 0, "updated": 0, "unchanged": 0, "empty": 0, "failed": 0}
    async with hc.AgentRun(AGENT, workspace=a.workspace,
                           detail={"root": str(root), "label": a.label,
                                   "files": len(docs)}) as run:
        for i, d in enumerate(docs, 1):
            try:
                text = d.text()
            except Exception as e:
                counts["failed"] += 1
                print(f"[{i}/{len(docs)}] FAILED  {d.title}  ({type(e).__name__}: {e})",
                      file=sys.stderr)
                continue

            res = await hc.store_document(
                doc_key=d.key,
                text=text,
                source="gdrive",
                workspace=a.workspace,
                meta={
                    "file_name": d.path.name,
                    "title": d.title,
                    "path": str(d.path.relative_to(root)),
                    "modified_at": d.modified,
                    "library": a.label or root.name,
                },
            )
            counts[res.status] = counts.get(res.status, 0) + 1
            run.wrote(res)
            run.fetched(1)
            print(f"[{i}/{len(docs)}] {res.status:<9} {res.written or res.skipped:>3}  {d.title[:58]}",
                  file=sys.stderr)

    print()
    print(f"Run {run.run_id}: " + " · ".join(f"{k} {v}" for k, v in counts.items() if v))
    print(f"{run.rows_written} chunks written into workspace '{a.workspace}'")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
