"""
findings — review what the agents proposed.

An agent records a finding; a human accepts or rejects it; an accepted finding
is written into the brain file it answers. Without that last step nothing
compounds — ten agent runs produce ten reports and an unchanged brain.

This half handles review and export. The brain files are in a different repo
that this machine deliberately cannot reach, so writing them is the other half:
`scripts/promote_findings.py` in csmart-gtm, which takes the JSON `export`
produces and needs nothing but the standard library.

    python agents/findings.py list
    python agents/findings.py show 2
    python agents/findings.py accept 2
    python agents/findings.py reject 3 --why "competitor named but pricing is an inference"
    python agents/findings.py export --status accepted > accepted.json
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_core as hc

WRAP = 78


def fetch(status=None, workspace=None, fid=None) -> list[dict]:
    q = hc.supabase.table("agent_findings").select("*").order("id")
    if fid:
        q = q.eq("id", fid)
    if status and status != "all":
        q = q.eq("status", status)
    if workspace:
        q = q.eq("workspace", workspace)
    return q.execute().data or []


def cmd_list(a) -> int:
    rows = fetch(a.status, a.workspace)
    if not rows:
        print(f"No findings with status '{a.status}'"
              + (f" in workspace '{a.workspace}'" if a.workspace else ""))
        return 0

    print(f"{'id':>4}  {'status':<9} {'conf':<10} {'workspace':<14} target / marker")
    print("  " + "-" * (WRAP - 2))
    for r in rows:
        print(f"{r['id']:>4}  {r['status']:<9} {r['confidence']:<10} "
              f"{r['workspace']:<14} {r['target']}")
        print(f"      {(r['marker'] or '')[:70]}")
        print(f"      {textwrap.shorten(r['statement'], 68)}")
    print()
    print(f"{len(rows)} finding(s). `show <id>` for the evidence.")
    return 0


def cmd_show(a) -> int:
    rows = fetch(fid=a.id)
    if not rows:
        print(f"No finding #{a.id}", file=sys.stderr)
        return 1
    r = rows[0]

    def field(k, v):
        print(f"  {k:<12} {v}")

    print(f"\nFinding #{r['id']}")
    print("  " + "-" * (WRAP - 2))
    field("status", r["status"])
    field("confidence", r["confidence"])
    field("source kind", r["source_kind"])
    field("workspace", r["workspace"])
    field("target", r["target"])
    field("marker", r["marker"] or "—")
    field("source", r["source_url"] or "—")
    field("fetched", r["fetched_at"] or "—")
    print()
    print("  Statement")
    for line in textwrap.wrap(r["statement"], WRAP - 6):
        print(f"      {line}")
    print()
    print("  Evidence — verbatim from the source, checked against it")
    for line in textwrap.wrap(r["evidence"] or "(none recorded)", WRAP - 8):
        print(f"      > {line}")
    print()
    if r["status"] == "proposed":
        print(f"  accept:  python agents/findings.py accept {r['id']}")
        print(f"  reject:  python agents/findings.py reject {r['id']} --why '...'")
    return 0


def _review(fid: int, status: str, why: str | None) -> int:
    rows = fetch(fid=fid)
    if not rows:
        print(f"No finding #{fid}", file=sys.stderr)
        return 1
    r = rows[0]
    if r["status"] != "proposed":
        print(f"Finding #{fid} is already '{r['status']}' — nothing to do.", file=sys.stderr)
        return 1

    patch = {"status": status, "reviewed_at": datetime.now(timezone.utc).isoformat()}
    if why:
        # Kept on the row rather than in a log: the reason a finding was
        # rejected is the most useful thing to have when a later run proposes
        # something similar.
        patch["evidence"] = (r["evidence"] or "") + f"\n\n[rejected: {why}]"
    hc.supabase.table("agent_findings").update(patch).eq("id", fid).execute()

    print(f"Finding #{fid} → {status}")
    print(f"  {textwrap.shorten(r['statement'], 70)}")
    if status == "accepted":
        print()
        print("  Not yet in the brain. Export and promote:")
        print("    python agents/findings.py export --status accepted > accepted.json")
        print("  then in csmart-gtm:")
        print("    python scripts/promote_findings.py accepted.json --dry-run")
    return 0


def cmd_accept(a):
    return _review(a.id, "accepted", None)


def cmd_reject(a):
    return _review(a.id, "rejected", a.why)


def cmd_export(a) -> int:
    rows = fetch(a.status, a.workspace)
    out = [{
        "id": r["id"],
        "workspace": r["workspace"],
        "target": r["target"],
        "marker": r["marker"],
        "statement": r["statement"],
        "evidence": r["evidence"],
        "source_url": r["source_url"],
        "source_kind": r["source_kind"],
        "confidence": r["confidence"],
        "fetched_at": r["fetched_at"],
        "reviewed_at": r["reviewed_at"],
    } for r in rows]
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    print(f"{len(out)} finding(s) exported", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list");   p.set_defaults(fn=cmd_list)
    p.add_argument("--status", default="proposed",
                   choices=["proposed", "accepted", "rejected", "all"])
    p.add_argument("--workspace", default="")

    p = sub.add_parser("show");   p.set_defaults(fn=cmd_show); p.add_argument("id", type=int)
    p = sub.add_parser("accept"); p.set_defaults(fn=cmd_accept); p.add_argument("id", type=int)
    p = sub.add_parser("reject"); p.set_defaults(fn=cmd_reject); p.add_argument("id", type=int)
    p.add_argument("--why", default="", help="why — worth recording, a later run may propose it again")

    p = sub.add_parser("export"); p.set_defaults(fn=cmd_export)
    p.add_argument("--status", default="accepted",
                   choices=["proposed", "accepted", "rejected", "all"])
    p.add_argument("--workspace", default="")

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
