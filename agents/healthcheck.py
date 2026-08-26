"""
healthcheck — prove the external dependencies still work, and say so loudly.

Written after nvidia/nv-embedqa-e5-v5 was retired at 09:00Z on 2026-08-25 and
nobody noticed for thirteen hours. Every check here corresponds to something
that has actually broken this system:

    embeddings   410 Gone — the model reached end of life mid-day
    openrouter   402 Payment Required — credit ran out mid-run
    search       returned nothing while looking healthy (the ivfflat index),
                 and would do so again on any model/corpus dimension mismatch
    supabase     the store everything else depends on
    bot          the service that survives reboots, until it doesn't

Search is the one that matters most. The others fail loudly; a retrieval system
that returns zero results looks exactly like a corpus with nothing relevant in
it, which is why the ivfflat fault survived weeks and the model retirement
survived a day.

Alerts go to Telegram. Notifies on transition — broken, then recovered — and
re-nags every NAG_HOURS while still broken, so a persistent fault neither spams
nor goes quiet.

    python agents/healthcheck.py            # quiet unless something is wrong
    python agents/healthcheck.py --verbose  # always print the table
    python agents/healthcheck.py --test-alert
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_core as hc

AGENT = "healthcheck"
STATE = Path(__file__).resolve().parent.parent / ".healthcheck-state.json"
NAG_HOURS = 6

# Below this, OpenRouter is close enough to empty to warn before it 402s.
LOW_BALANCE_USD = 5.0

# A phrase that must be findable if retrieval works at all. Chosen because it
# sits in the Vantage corpus; update it if that document ever leaves.
CANARY_QUERY = "non-technical losses in utility billing"
CANARY_MIN_HITS = 1


class Check:
    def __init__(self, name):
        self.name, self.ok, self.detail = name, False, ""

    def passed(self, detail=""):
        self.ok, self.detail = True, detail
        return self

    def failed(self, detail):
        self.ok, self.detail = False, detail
        return self


async def check_embeddings() -> Check:
    c = Check("embeddings")
    try:
        v = await hc.embed("healthcheck", "query")
        if len(v) != hc.EMBED_DIMS:
            return c.failed(f"{hc.EMBED_MODEL} returned {len(v)} dims, expected {hc.EMBED_DIMS}")
        return c.passed(f"{hc.EMBED_MODEL} · {len(v)} dims")
    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = e.response.json().get("detail", "")[:120]
        except Exception:
            body = e.response.text[:120]
        return c.failed(f"HTTP {e.response.status_code} — {body}")
    except Exception as e:
        return c.failed(f"{type(e).__name__}: {e}")


async def check_openrouter() -> Check:
    """Balance only — free to call, and catches the 402 before a run hits it."""
    c = Check("openrouter")
    try:
        r = httpx.get("https://openrouter.ai/api/v1/credits",
                      headers={"Authorization": f"Bearer {hc.OPENROUTER_KEY}"}, timeout=20)
        r.raise_for_status()
        d = r.json()["data"]
        bal = float(d["total_credits"]) - float(d["total_usage"])
        if bal <= 0:
            return c.failed(f"balance ${bal:.2f} — smart tier will 402")
        if bal < LOW_BALANCE_USD:
            return c.failed(f"balance ${bal:.2f} — below ${LOW_BALANCE_USD:.0f}, top up")
        return c.passed(f"${bal:.2f} remaining")
    except Exception as e:
        return c.failed(f"{type(e).__name__}: {e}")


async def check_fast_tier() -> Check:
    c = Check("nim chat")
    try:
        r = await hc.chat([{"role": "user", "content": "Reply with the single word: ok"}],
                          tier="fast", max_tokens=600, strict=False)
        return c.passed(f"{hc.FAST_MODEL.split('/')[-1]} · {r.output_tokens} tok")
    except Exception as e:
        return c.failed(f"{type(e).__name__}: {str(e)[:110]}")


async def check_supabase() -> Check:
    c = Check("supabase")
    try:
        rows = hc.supabase.table("memories").select("id").execute().data
        nulls = hc.supabase.table("memories").select("id").is_("embedding", "null").execute().data
        if nulls:
            return c.failed(f"{len(nulls)} of {len(rows)} rows have no embedding")
        return c.passed(f"{len(rows)} rows, all embedded")
    except Exception as e:
        return c.failed(f"{type(e).__name__}: {str(e)[:110]}")


async def check_search() -> Check:
    """The check that matters. A retrieval system returning nothing looks
    identical to one with nothing relevant to return."""
    c = Check("search")
    try:
        hits = await hc.search(CANARY_QUERY, limit=5)
        if len(hits) < CANARY_MIN_HITS:
            return c.failed(f"canary query returned {len(hits)} hits — retrieval is broken "
                            f"or the corpus no longer contains the canary document")
        return c.passed(f"{len(hits)} hits, top {hits[0]['similarity']:.3f}")
    except Exception as e:
        return c.failed(f"{type(e).__name__}: {str(e)[:110]}")


def check_bot() -> Check:
    c = Check("bot service")
    try:
        out = subprocess.run(["systemctl", "is-active", "hermes"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return c.passed("active") if out == "active" else c.failed(f"systemctl says '{out}'")
    except Exception as e:
        return c.failed(f"{type(e).__name__}: {e}")


# ---------- alerting ----------

def telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    ids = [u.strip() for u in os.getenv("ALLOWED_USER_ID", "").split(",") if u.strip()]
    if not token or not ids:
        return False
    sent = False
    for uid in ids:
        try:
            r = httpx.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=20,
                           json={"chat_id": uid, "text": text, "disable_web_page_preview": True})
            sent = sent or r.status_code == 200
        except Exception:
            pass
    return sent


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"failing": False, "last_alert": 0}


def save_state(d: dict) -> None:
    try:
        STATE.write_text(json.dumps(d))
    except Exception:
        pass


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="print the table even when healthy")
    ap.add_argument("--no-alert", action="store_true")
    ap.add_argument("--test-alert", action="store_true", help="send a test message and exit")
    a = ap.parse_args()

    if a.test_alert:
        ok = telegram("Hermes healthcheck: test alert. If you can read this, alerting works.")
        print("sent" if ok else "FAILED to send — check TELEGRAM_BOT_TOKEN / ALLOWED_USER_ID")
        return 0 if ok else 1

    checks = [await check_embeddings(), await check_openrouter(), await check_fast_tier(),
              await check_supabase(), await check_search(), check_bot()]
    bad = [c for c in checks if not c.ok]

    if a.verbose or bad:
        for c in checks:
            print(f"  {'ok  ' if c.ok else 'FAIL'} {c.name:<12} {c.detail}")

    state = load_state()
    now = time.time()
    should_alert = bool(bad) and (
        not state["failing"] or now - state.get("last_alert", 0) > NAG_HOURS * 3600)

    if bad and should_alert and not a.no_alert:
        lines = ["⚠️ Hermes healthcheck FAILED", ""]
        lines += [f"✗ {c.name}: {c.detail}" for c in bad]
        healthy = [c.name for c in checks if c.ok]
        if healthy:
            lines += ["", "still ok: " + ", ".join(healthy)]
        telegram("\n".join(lines))

    if not bad and state["failing"] and not a.no_alert:
        telegram("✅ Hermes healthcheck recovered — all checks passing again.")

    # --no-alert is a test mode; persisting from it would make the next real
    # run announce a recovery from a failure that was deliberately induced.
    if not a.no_alert:
        save_state({"failing": bool(bad),
                    "last_alert": now if should_alert else state.get("last_alert", 0)})

    # Only record a run when something is wrong. An hourly check that logs
    # every pass buries the failures it exists to surface.
    if bad:
        try:
            async with hc.AgentRun(AGENT, detail={"failed": [c.name for c in bad],
                                                  "detail": {c.name: c.detail for c in bad}}):
                raise RuntimeError("; ".join(f"{c.name}: {c.detail}" for c in bad))
        except RuntimeError:
            pass

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
