"""
Exercise hermes_core against the live database, then clean up after itself.

Run on csmart-one after a migration:
    cd ~/hermes && venv/bin/python smoke_test.py

Proves the things that are easy to believe and hard to verify: that
re-ingesting an unchanged document writes nothing, that a changed one
replaces rather than accumulates, and that AgentRun closes its row.
"""

import asyncio
import sys

import hermes_core as hc

DOC = "smoke:hermes_core"
AGENT = "smoke-test"

TEXT_A = ("Hermes smoke test alpha. " * 30) + \
         "The distinguishing phrase for run alpha is periwinkle telescope."
TEXT_B = ("Hermes smoke test beta. " * 30) + \
         "The distinguishing phrase for run beta is marmalade lighthouse."

failures = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def cleanup() -> None:
    hc.supabase.table("memories").delete().eq("doc_key", DOC).execute()
    hc.supabase.table("agent_runs").delete().eq("agent", AGENT).execute()


async def main() -> None:
    cleanup()

    # --- first ingest ---
    r1 = await hc.store_document(DOC, TEXT_A, "test", {"title": "Smoke A"})
    check("first ingest stores chunks", r1.status == "stored" and r1.written > 0, str(r1))

    # --- identical re-ingest writes nothing ---
    r2 = await hc.store_document(DOC, TEXT_A, "test", {"title": "Smoke A"})
    check("unchanged re-ingest is a no-op", r2.status == "unchanged" and r2.written == 0, str(r2))

    rows = hc.supabase.table("memories").select("id").eq("doc_key", DOC).execute().data
    check("no duplicate rows accumulated", len(rows) == r1.written,
          f"{len(rows)} rows in DB vs {r1.written} written")

    # --- changed document replaces, does not accumulate ---
    r3 = await hc.store_document(DOC, TEXT_B, "test", {"title": "Smoke B"})
    check("changed document is replaced", r3.status == "updated" and r3.written > 0, str(r3))

    rows = hc.supabase.table("memories").select("content").eq("doc_key", DOC).execute().data
    check("old version gone after replace", len(rows) == r3.written,
          f"{len(rows)} rows in DB vs {r3.written} written")
    check("stored content is the new version",
          any("marmalade lighthouse" in r["content"] for r in rows))
    check("old content is absent",
          not any("periwinkle telescope" in r["content"] for r in rows))

    # --- retrieval reaches it ---
    hits = await hc.search("marmalade lighthouse smoke test", limit=5)
    check("semantic search finds the document",
          any(h.get("doc_key") == DOC or "marmalade" in h["content"] for h in hits),
          f"{len(hits)} hits")

    # --- run logging ---
    async with hc.AgentRun(AGENT, spend_cap_usd=1.0) as run:
        run.fetched(2)
        run.wrote(r3)

    logged = (hc.supabase.table("agent_runs").select("*")
              .eq("agent", AGENT).order("started_at", desc=True).limit(1).execute().data)
    check("agent run logged and closed",
          bool(logged) and logged[0]["status"] == "ok"
          and logged[0]["finished_at"] is not None
          and logged[0]["sources_fetched"] == 2,
          str(logged[0]) if logged else "no row")

    # --- spend cap halts rather than crashes ---
    async with hc.AgentRun(AGENT, spend_cap_usd=-1.0) as run:
        run.check_cap()
        check("spend cap raised", False, "check_cap did not raise")

    halted = (hc.supabase.table("agent_runs").select("status,error")
              .eq("agent", AGENT).order("started_at", desc=True).limit(1).execute().data)
    check("over-cap run recorded as halted",
          bool(halted) and halted[0]["status"] == "halted", str(halted))

    cleanup()
    left = hc.supabase.table("memories").select("id").eq("doc_key", DOC).execute().data
    check("cleanup removed test rows", len(left) == 0)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
