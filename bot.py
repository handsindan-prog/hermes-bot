"""
Hermes Telegram bot — the capture surface.

Everything reusable (extraction, chunking, embedding, storage, retrieval,
reasoning) lives in hermes_core so agents share it. This file is only the
Telegram wiring.
"""

import logging
import os
import tempfile

import trafilatura
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import hermes_core as hc
from hermes_core import EXTRACTORS, URL_RE, clean_text, normalise_url

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# httpx logs request URLs at INFO, which prints the bot token. Leaked it three
# times before this line existed.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("hermes")

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

# Comma-separated list of Telegram user IDs. A single bare ID still works, so
# existing .env files keep behaving exactly as before.
#
# This has to be a set compared member-wise, not a string compared whole: the
# obvious "append another id" edit against the old `== ALLOWED` check locked
# everyone out, because no single user's id equals "id1,id2".
ALLOWED = {
    uid.strip()
    for uid in os.getenv("ALLOWED_USER_ID", "").split(",")
    if uid.strip()
}


# ---------- handlers ----------

def authorised(update: Update) -> bool:
    # An empty allow-list means the bot is open. That is the historical
    # behaviour and is only ever right for local testing.
    if not ALLOWED:
        return True
    return str(update.effective_user.id) in ALLOWED


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Your Telegram user ID is: {update.effective_user.id}")


async def recall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorised(update):
        return
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text("Usage: /recall search terms")
        return
    try:
        rows = await hc.search(query, limit=5)
        if not rows:
            await update.message.reply_text("Nothing relevant found.")
            return
        lines = [
            f"[{r['created_at'][:10]} · {hc.label_of(r)} · {round(r['similarity'], 2)}]\n"
            f"{r['content'][:400]}"
            for r in rows
        ]
        await update.message.reply_text("\n\n".join(lines)[:4000])
    except Exception as e:
        log.exception("Recall failed")
        await update.message.reply_text(f"Recall failed: {e}")


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorised(update):
        return
    question = " ".join(context.args).strip()
    if not question:
        await update.message.reply_text("Usage: /ask your question")
        return
    await update.message.chat.send_action("typing")
    try:
        rows = await hc.search(question)
        result = await hc.reason(question, rows)
        answer = result.text[:4000]
        if result.truncated:
            answer += "\n\n(cut off at the length limit — ask something narrower)"
        await update.message.reply_text(answer)
    except Exception as e:
        log.exception("Ask failed")
        await update.message.reply_text(f"Ask failed: {e}")


async def runs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Last few agent runs — the read side of the agent_runs log."""
    if not authorised(update):
        return
    try:
        rows = (
            hc.supabase.table("agent_runs")
            .select("agent,status,started_at,rows_written,cost_usd,error")
            .order("started_at", desc=True)
            .limit(10)
            .execute()
            .data
        )
        if not rows:
            await update.message.reply_text("No agent runs logged yet.")
            return
        lines = []
        for r in rows:
            line = (f"{r['started_at'][:16].replace('T', ' ')} · {r['agent']} · "
                    f"{r['status']} · {r['rows_written']} rows · ${float(r['cost_usd']):.4f}")
            if r.get("error"):
                line += f"\n   {r['error'][:180]}"
            lines.append(line)
        await update.message.reply_text("\n".join(lines)[:4000])
    except Exception as e:
        log.exception("Runs failed")
        await update.message.reply_text(f"Runs failed: {e}")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorised(update):
        return

    doc = update.message.document
    name = doc.file_name or "unnamed"
    ext = os.path.splitext(name)[1].lower()

    if ext not in EXTRACTORS:
        await update.message.reply_text(
            f"Can't read {ext or 'that'}. Supported: pdf, docx, pptx, txt, md"
        )
        return

    await update.message.reply_text(f"Reading {name}...")
    await update.message.chat.send_action("typing")

    try:
        tg_file = await context.bot.get_file(doc.file_id)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
            await tg_file.download_to_drive(tmp.name)
            text = EXTRACTORS[ext](tmp.name)

        result = await hc.store_document(
            doc_key=name,
            text=text,
            source=ext.lstrip("."),
            meta={
                "file_name": name,
                "title": os.path.splitext(name)[0],
                "user_id": update.effective_user.id,
            },
        )

        if result.status == "empty":
            await update.message.reply_text("No readable text found — is it a scanned image?")
        elif result.status == "unchanged":
            await update.message.reply_text(f"{name} is already stored and unchanged.")
        elif result.status == "updated":
            await update.message.reply_text(
                f"{name} changed — replaced with {result.written} chunks."
            )
        else:
            await update.message.reply_text(f"Stored {name} as {result.written} chunks.")

    except Exception as e:
        log.exception("Document ingest failed")
        await update.message.reply_text(f"Failed: {e}")


async def ingest_url(update: Update, url: str):
    await update.message.reply_text(f"Fetching {url[:60]}...")
    await update.message.chat.send_action("typing")

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            await update.message.reply_text(f"Couldn't fetch {url[:60]}")
            return

        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        meta = trafilatura.extract_metadata(downloaded)
        title = (meta.title if meta else None) or url

        if not text:
            await update.message.reply_text("No readable content extracted.")
            return

        result = await hc.store_document(
            doc_key=url,
            text=text,
            source="web",
            meta={
                "url": url,
                "title": clean_text(title)[:200],
                "user_id": update.effective_user.id,
            },
        )

        if result.status == "unchanged":
            await update.message.reply_text(f'"{title[:80]}" unchanged since last fetch.')
        elif result.status == "updated":
            await update.message.reply_text(
                f'"{title[:80]}" changed — replaced with {result.written} chunks.'
            )
        else:
            await update.message.reply_text(f'Stored "{title[:80]}" as {result.written} chunks.')

    except Exception as e:
        log.exception("URL ingest failed")
        await update.message.reply_text(f"Failed: {e}")


async def capture(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorised(update):
        user = update.effective_user
        log.warning("Rejected message from %s (@%s, %s)",
                    user.id, user.username, user.full_name)
        return

    text = clean_text(update.message.text)
    urls = [normalise_url(u) for u in URL_RE.findall(text)]

    try:
        if urls:
            for url in urls[:3]:
                await ingest_url(update, url)

            # if there's real commentary alongside the link, keep that too
            remainder = URL_RE.sub("", text).strip()
            if len(remainder.split()) >= 5:
                await hc.store_note(text, {
                    "user_id": update.effective_user.id,
                    "chat_id": update.effective_chat.id,
                    "message_id": update.message.message_id,
                })
                await update.message.reply_text("Note saved alongside.")
            return

        await hc.store_note(text, {
            "user_id": update.effective_user.id,
            "chat_id": update.effective_chat.id,
            "message_id": update.message.message_id,
        })
        await update.message.reply_text("Saved.")

    except Exception as e:
        log.exception("Capture failed")
        await update.message.reply_text(f"Failed to save: {e}")


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("recall", recall))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("runs", runs))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, capture))
    log.info("Hermes starting (polling)")
    app.run_polling()


if __name__ == "__main__":
    main()
