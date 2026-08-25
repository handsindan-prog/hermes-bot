"""
reembed — regenerate every stored vector with the current embedding model.

Needed whenever EMBED_MODEL changes. Vectors from different models occupy
different spaces and are not comparable, so a model change invalidates the
whole corpus at once — there is no partial migration and no mixing.

`content` is the source of truth and is never touched, so this is repeatable
and safe to interrupt: it only ever fills in rows whose embedding is null,
and re-running picks up where it stopped.

    python agents/reembed.py --dry-run
    python agents/reembed.py
    python agents/reembed.py --workspace vantage      # one workspace at a time
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_core as hc

AGENT = "reembed"
BATCH = 32          # rows fetched per round; embed_batch chunks to 16 internally


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="", help="limit to one workspace")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    def pending(limit=None):
        q = (hc.supabase.table("memories").select("id,content")
             .is_("embedding", "null").order("id"))
        if a.workspace:
            q = q.eq("workspace", a.workspace)
        if limit:
            q = q.limit(limit)
        return q.execute().data or []

    todo = pending()
    total_rows = len(hc.supabase.table("memories").select("id").execute().data)

    print(f"model      {hc.EMBED_MODEL} ({hc.EMBED_DIMS} dims)")
    print(f"corpus     {total_rows} rows")
    print(f"to embed   {len(todo)}" + (f"  (workspace: {a.workspace})" if a.workspace else ""))

    if not todo:
        print("\nNothing pending — every row already has a vector.")
        return 0
    if a.dry_run:
        print("\nDry run — nothing written.")
        return 0

    done = failed = 0
    async with hc.AgentRun(AGENT, workspace=a.workspace or hc.WORKSPACE,
                           detail={"model": hc.EMBED_MODEL, "dims": hc.EMBED_DIMS,
                                   "pending": len(todo)}) as run:
        while True:
            batch = pending(BATCH)
            if not batch:
                break
            try:
                vectors = await hc.embed_batch([r["content"] for r in batch], "passage")
            except Exception as e:
                # A whole batch failing is an API problem, not a data problem —
                # stop rather than grinding through every row to fail identically.
                print(f"\nembedding failed: {type(e).__name__}: {e}", file=sys.stderr)
                failed = len(batch)
                raise

            for row, vec in zip(batch, vectors):
                hc.supabase.table("memories").update(
                    {"embedding": vec}).eq("id", row["id"]).execute()
                done += 1
            run.wrote(len(batch))
            print(f"  {done}/{len(todo)} re-embedded", file=sys.stderr)

    print(f"\nRun {run.run_id}: {done} rows re-embedded with {hc.EMBED_MODEL}")
    if failed:
        print(f"{failed} left pending — re-run to continue")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
