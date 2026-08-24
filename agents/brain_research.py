"""
brain_research — work the [NEEDS RESEARCH] markers in a GTM brain.

For each open marker it searches the Hermes corpus, and where the retrieved
passages actually answer the question it proposes a claim with the verbatim
text that claim rests on. Where they don't, it says so and states what would
answer it. Nothing is written to the brain and nothing is written to memories:
every output is a row in agent_claims, awaiting a human.

The rule this exists to obey:

    Never let the model be the source of facts about entities. A tool
    retrieves; the model ranks, classifies, summarises and drafts.

That is enforced rather than requested. The model is only ever shown retrieved
passages and asked whether they answer the question, and any quote it returns
is checked to appear verbatim in the passage it cited. A claim whose quote
cannot be found is discarded, because a model that invents a quote will invent
a company name.

Usage:
    venv/bin/python agents/brain_research.py --brain ./gtm --workspace vantage
    venv/bin/python agents/brain_research.py --brain ./gtm --workspace vantage \
        --also-search circularsmart --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# hermes_core sits one level up; without this, running the script by path
# puts agents/ on sys.path instead of the repo root and the import fails.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_core as hc

AGENT = "brain-research"
MARKER = "[NEEDS RESEARCH]"
HEADING = re.compile(r"^#{1,6}\s")

# Retrieval is deliberately generous and judgement deliberately strict: better
# to show the model six passages and have it reject five than to miss the one
# passage that answers the question.
SEARCH_LIMIT = 6
SEARCH_THRESHOLD = 0.20

JUDGE_SYSTEM = """You decide whether retrieved passages answer a research question.

You are not a source of facts. You have no knowledge of these companies,
products or markets beyond the passages given to you. If the passages do not
answer the question, the answer is NO — that is a useful, expected result and
you will not be penalised for it.

Never state a company name, number, date or claim that does not appear in the
passages. Never infer a fact that the passages merely make plausible.

Reply with JSON only, no prose and no code fence:

{
  "answered": true|false,
  "claim": "one sentence stating what the passages establish",
  "passage": <the number of the single passage the claim rests on>,
  "evidence": "a verbatim span copied exactly from that passage",
  "confidence": "high"|"medium"|"low",
  "missing": "if answered is false, what a source would need to state"
}

"evidence" must be copied character-for-character from the passage. Do not
tidy it, shorten mid-sentence with ellipses, or fix its punctuation — it is
checked against the original and a claim whose quote cannot be found is thrown
away.

confidence: high if a passage states it outright; medium if it states it
partially or in passing; low if it is only implied."""


@dataclass
class Marker:
    file: str
    line: int
    title: str
    body: str

    @property
    def question(self) -> str:
        return f"{self.title}. {self.body}".strip()

    def __str__(self) -> str:
        return f"{self.file}:{self.line}  {self.title}"


def parse_markers(root: Path) -> list[Marker]:
    """Every [NEEDS RESEARCH] marker, with the prose under it as context."""
    out: list[Marker] = []
    for path in sorted(root.rglob("*.md")):
        rel = str(path.relative_to(root))
        # The convention is documented in these files; the markers there are
        # explanations of the convention, not open questions.
        if rel in {"CLAUDE.md", "README.md"} or "/skills/" in rel or rel.startswith("skills/"):
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, line in enumerate(lines):
            if MARKER not in line:
                continue
            title = line.split(MARKER, 1)[1].strip(" #:-") or "(untitled)"
            body: list[str] = []
            for nxt in lines[i + 1:]:
                if HEADING.match(nxt) or MARKER in nxt:
                    break
                body.append(nxt)
            out.append(Marker(rel, i + 1, title, " ".join(body).strip()))
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def quote_is_real(quote: str, passages: list[dict]) -> tuple[bool, str]:
    """Is this quote actually present in one of the passages we retrieved?

    The single most important check in the file. A model that will fabricate a
    quote will fabricate a company name, and the corpus cannot tell the
    difference later.
    """
    q = _norm(quote)
    if len(q) < 12:
        return False, "quote too short to verify"
    for p in passages:
        if q in _norm(p["content"]):
            return True, p.get("doc_key") or hc.label_of(p)
    return False, "quote not found in any retrieved passage"


def parse_json(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


async def investigate(m: Marker, workspaces: list[str], run: hc.AgentRun,
                      tier: str = "smart") -> dict:
    """Search, judge, verify. Returns a result dict for the report."""
    passages: list[dict] = []
    for ws in workspaces:
        passages += await hc.search(m.question, limit=SEARCH_LIMIT,
                                    threshold=SEARCH_THRESHOLD, workspace=ws)
    run.fetched(len(passages))

    if not passages:
        return {"marker": m, "status": "no-corpus",
                "missing": "Nothing in the corpus is even topically close. "
                           "This needs an external source ingested first."}

    # Deduplicate by id, keep the strongest, cap what we pay to reason over.
    seen, uniq = set(), []
    for p in sorted(passages, key=lambda x: -x["similarity"]):
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        uniq.append(p)
    uniq = uniq[:SEARCH_LIMIT]

    listing = "\n\n".join(
        f"[{i + 1}] (workspace: {p.get('workspace')}, source: {p['source']}, "
        f"document: {hc.label_of(p)}, similarity: {p['similarity']:.3f})\n{p['content']}"
        for i, p in enumerate(uniq)
    )

    res = await hc.chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content":
                f"Research question, from {m.file}:\n\n{m.question}\n\n"
                f"---\n\nRetrieved passages:\n\n{listing}"},
        ],
        # Nemotron bills its reasoning against max_tokens and returns its own
        # chain-of-thought if starved, so the fast tier gets real headroom.
        tier=tier, max_tokens=1800 if tier == "fast" else 700,
    )
    run.charge(res)

    verdict = parse_json(res.text)
    if verdict is None:
        return {"marker": m, "status": "unparseable", "raw": res.text[:300]}

    if not verdict.get("answered"):
        return {"marker": m, "status": "no-evidence",
                "missing": verdict.get("missing") or "(model gave no reason)",
                "best": uniq[0]}

    quote = verdict.get("evidence", "")
    ok, where = quote_is_real(quote, uniq)
    if not ok:
        # Treated as a failure of the run, not a soft claim. This is the
        # fabrication case and it should be loud.
        return {"marker": m, "status": "quote-unverifiable",
                "claim": verdict.get("claim", ""), "quote": quote, "why": where}

    idx = verdict.get("passage")
    cited = uniq[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(uniq) else uniq[0]
    meta = cited.get("metadata") or {}
    url = meta.get("url") or ""

    claim = hc.Claim(
        target=m.file,
        marker=f"{MARKER} {m.title}",
        claim=verdict.get("claim", "").strip(),
        evidence=quote.strip(),
        source_url=url or f"corpus:{where}",
        source_kind="corpus",
        confidence=verdict.get("confidence", "low"),
        fetched_at=(cited.get("created_at") or "")[:10] or None,
    )
    claim_id = run.claim(claim)
    return {"marker": m, "status": "claim", "claim": claim,
            "claim_id": claim_id, "cited": cited}


def report(results: list[dict], workspaces: list[str], run: hc.AgentRun) -> str:
    by = lambda s: [r for r in results if r["status"] == s]
    claims, noev, nocorp = by("claim"), by("no-evidence"), by("no-corpus")
    bad, unp = by("quote-unverifiable"), by("unparseable")

    L = [
        f"# Brain research — {', '.join(workspaces)}",
        "",
        f"Run {run.run_id} · {len(results)} markers · "
        f"{run.sources_fetched} passages retrieved · ${run.cost_usd:.4f}",
        "",
        f"- **{len(claims)}** claims proposed (awaiting review)",
        f"- **{len(noev)}** retrieved something, but nothing that answers the question",
        f"- **{len(nocorp)}** nothing in the corpus is even close",
    ]
    if bad:
        L.append(f"- **{len(bad)}** discarded — quote could not be verified against the source")
    if unp:
        L.append(f"- **{len(unp)}** unparseable model output")

    if claims:
        L += ["", "## Proposed claims", "",
              "Each rests on a verbatim quote checked against the retrieved passage.",
              "Accept or reject in `agent_claims`; nothing has touched the brain.", ""]
        for r in claims:
            c = r["claim"]
            L += [f"### {c.marker}",
                  f"*{r['marker'].file}* · confidence **{c.confidence}** · claim #{r['claim_id']}",
                  "", f"**Claim.** {c.claim}", "",
                  f"> {c.evidence}", "",
                  f"Source: `{c.source_url}` ({hc.label_of(r['cited'])}, "
                  f"similarity {r['cited']['similarity']:.3f})", ""]

    if noev or nocorp:
        L += ["", "## Still open", "",
              "The corpus cannot answer these. Each line names what would.", ""]
        for r in noev + nocorp:
            L.append(f"- **{r['marker'].file}** — {r['marker'].title}  \n  {r['missing']}")

    if bad:
        L += ["", "## Discarded — unverifiable quotes", "",
              "The model asserted a quote that is not in any passage it was shown. "
              "Treat these as a warning about the run, not as findings.", ""]
        for r in bad:
            L += [f"- **{r['marker'].file}** — {r['marker'].title}",
                  f"  - claimed: {r['claim']}",
                  f"  - quote: `{r['quote'][:120]}`",
                  f"  - {r['why']}"]

    return "\n".join(L)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True, help="path to the brain repo")
    ap.add_argument("--workspace", required=True, help="workspace to search and file claims under")
    ap.add_argument("--also-search", nargs="*", default=[],
                    help="additional workspaces to search (claims still file under --workspace)")
    ap.add_argument("--subdir", default="", help="limit to a subdirectory of brain/, e.g. vantage")
    ap.add_argument("--tier", choices=("smart", "fast"), default="smart",
                    help="smart = Claude via OpenRouter; fast = Nemotron via NVIDIA NIM. "
                         "The verbatim-quote check applies either way, so a weaker judge "
                         "can propose a weak claim but cannot invent a source.")
    ap.add_argument("--cap", type=float, default=0.50, help="spend cap in USD")
    ap.add_argument("--limit", type=int, default=0, help="stop after N markers (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="list markers and exit")
    ap.add_argument("--out", default="", help="write the report here as well as stdout")
    a = ap.parse_args()

    root = Path(a.brain).expanduser().resolve()
    scope = root / "brain" / a.subdir if a.subdir else root
    if not scope.exists():
        print(f"no such path: {scope}", file=sys.stderr)
        return 2

    markers = parse_markers(scope)
    if a.limit:
        markers = markers[:a.limit]

    if not markers:
        print(f"No {MARKER} markers under {scope}")
        return 0

    if a.dry_run:
        print(f"{len(markers)} markers under {scope}:\n")
        for m in markers:
            print(" ", m)
        return 0

    workspaces = [a.workspace] + [w for w in a.also_search if w != a.workspace]
    results = []

    async with hc.AgentRun(AGENT, workspace=a.workspace, spend_cap_usd=a.cap,
                           detail={"brain": str(scope), "searched": workspaces, "tier": a.tier}) as run:
        for i, m in enumerate(markers, 1):
            print(f"[{i}/{len(markers)}] {m}", file=sys.stderr)
            results.append(await investigate(m, workspaces, run, tier=a.tier))
            run.check_cap()

        text = report(results, workspaces, run)

    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"\n(written to {a.out})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
