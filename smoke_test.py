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
    hc.supabase.table("agent_findings").delete().eq("target", DOC).execute()
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

    # --- workspaces keep products apart ---
    r_a = await hc.store_document(DOC, TEXT_A, "test", {"title": "ws A"}, workspace="smoke-a")
    r_b = await hc.store_document(DOC, TEXT_B, "test", {"title": "ws B"}, workspace="smoke-b")
    check("same doc_key coexists in two workspaces",
          r_a.status == "stored" and r_b.status == "stored",
          f"{r_a} / {r_b}")

    rows_a = (hc.supabase.table("memories").select("content")
              .eq("workspace", "smoke-a").eq("doc_key", DOC).execute().data)
    check("workspace A untouched by write to B",
          any("periwinkle telescope" in r["content"] for r in rows_a))

    hits = await hc.search("marmalade lighthouse", limit=10, workspace="smoke-b")
    check("search scoped to a workspace returns only that workspace",
          bool(hits) and all(h.get("workspace") == "smoke-b" for h in hits),
          f"{len(hits)} hits, workspaces={sorted({h.get('workspace') for h in hits})}")

    # --- findings are proposals, never facts ---
    async with hc.AgentRun(AGENT, workspace="smoke-a") as run:
        fid = run.finding(hc.Finding(
            target=DOC, marker="NEEDS RESEARCH: test",
            statement="Telecoms is the nearest adjacency.",
            evidence="telecoms, commercial waste, and managed print are the three nearest adjacencies",
            source_url="https://example.com/brief", source_kind="corpus",
            confidence="high", fetched_at="2026-08-24",
        ))
    check("finding recorded against the run", fid > 0)

    open_ = hc.open_findings(workspace="smoke-a", target=DOC)
    check("finding is 'proposed', not fact", bool(open_) and open_[0]["status"] == "proposed")

    model_finding = hc.Finding(target=DOC, statement="Acme raised $40M.",
                               source_kind="model", confidence="high")
    check("a model-sourced finding is forced to unverified",
          model_finding.confidence == "unverified",
          f"got {model_finding.confidence!r}")

    try:
        hc.Finding(target=DOC, statement="x", confidence="pretty sure")
        check("invalid confidence rejected", False, "no error raised")
    except ValueError:
        check("invalid confidence rejected", True)

    for ws in ("smoke-a", "smoke-b"):
        hc.supabase.table("memories").delete().eq("workspace", ws).execute()

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
