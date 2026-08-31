import logging
import os
import re
import time

from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

known_group_ids = set()


# =========================================================
# OPENAI CONFIG
# =========================================================

OPENAI_MODEL = "gpt-4o-mini"

DAILY_BUDGET_USD = 0.20

INPUT_PRICE_PER_MILLION = 0.15
OUTPUT_PRICE_PER_MILLION = 0.60


# =========================================================
# DAILY USAGE
# =========================================================

daily_input_tokens = 0
daily_output_tokens = 0
current_day = time.strftime("%Y-%m-%d")


def reset_daily_usage_if_needed():
    global daily_input_tokens
    global daily_output_tokens
    global current_day

    today = time.strftime("%Y-%m-%d")

    if today != current_day:
        current_day = today
        daily_input_tokens = 0
        daily_output_tokens = 0

        logger.info("Günlük API kullanım sayacı sıfırlandı.")


def calculate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    return input_cost + output_cost


def current_daily_cost():
    return calculate_cost(
        daily_input_tokens,
        daily_output_tokens,
    )


# =========================================================
# PROMPTS
# =========================================================

TRANSLATION_PROMPT = """
Sen C2 seviyesinde ana dil yetkinliğine sahip, üst düzey profesyonel bir çevirmensin. Asla açıklama ekleme, yorum yapma, selamlama.
Görevin: Gelen metnin kaynak dilini tespit et ve hedef diller olan TÜRKÇE ve RUSÇA'ya en doğru, bağlamsal, akıcı ve profesyonel şekilde çevir.

Gelen metin TÜRKÇE ise:
🇷🇺 [Rusça çevirisi]

Gelen metin RUSÇA ise:
🇹🇷 [Türkçe çevirisi]

Gelen metin farklı bir dilde (örneğin İngilizce vb.) ise:
🇹🇷 [Türkçe çevirisi]
🇷🇺 [Rusça çevirisi]

Kural: Sadece bayrak emojisi ve profesyonel çeviriyi yaz. Kelimesi kelimesine değil, anlam bütünlüğünü koruyan C2 seviyesinde profesyonel bir dil kullan.
"""

ASSISTANT_PROMPT = """
Sen son derece zeki, analitik, bilge ve üst düzey profesyonel bir yapay zeka asistanısın. Kullanıcıların sorularına net, doğru, etkileyici, akıllıca ve kapsamlı yanıtlar ver.
"""


# =========================================================
# OPENAI CLIENT
# =========================================================

client = None


def get_openai_client():
    global client
    if client is None:
        client = OpenAI(api_key=OPENAI_API_KEY)
    return client


# =========================================================
# CLEAN RESPONSE
# =========================================================

def clean_response(text):
    if not text:
        return ""

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = re.sub(
        r"\([^)]*note[^)]*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\(Note:.*?\)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return text.strip()


# =========================================================
# OPENAI REQUEST HELPER
# =========================================================

def ask_openai(system_prompt, user_text):
    global daily_input_tokens
    global daily_output_tokens

    reset_daily_usage_if_needed()

    current_cost = current_daily_cost()

    if current_cost >= DAILY_BUDGET_USD:
        logger.warning("Günlük bütçe sınırına ulaşıldı: $%.6f", current_cost)
        return None

    try:
        openai_client = get_openai_client()

        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_text,
                },
            ],
            max_tokens=300,
            temperature=0.3,
        )

        usage = getattr(response, "usage", None)

        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)

            daily_input_tokens += input_tokens
            daily_output_tokens += output_tokens

            logger.info(
                "API usage | input=%s | output=%s | daily_cost=$%.6f",
                input_tokens,
                output_tokens,
                current_daily_cost(),
            )

        if not response.choices:
            return ""

        output = response.choices[0].message.content

        return clean_response(output)

    except Exception as e:
        logger.error("OpenAI Error: %s", e, exc_info=True)
        return "⚠️ HATA (debug): " + str(e)


# =========================================================
# COMMAND HANDLERS
# =========================================================

async def about_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    about_text = (
        "🤖 Viyana AI\n\n"
        "Ben EHED tarafından oluşturulmuş C2 düzeyinde profesyonel "
        "Rusça-Türkçe çeviri ve zeki asistan botuyum."
    )
    await update.effective_message.reply_text(about_text)


# =========================================================
# MESSAGE HANDLER
# =========================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message

    if not message or not message.text:
        return

    if update.effective_chat.type in ["group", "supergroup"]:
        known_group_ids.add(update.effective_chat.id)

    text = message.text.strip()

    if update.effective_chat.type == "private" and text.startswith("02021995"):
        text_to_send = text[8:].strip()

        if not text_to_send:
            await message.reply_text("⚠️ Lütfen mesaj metnini yazın.")
            return

        if not known_group_ids:
            await message.reply_text("⚠️ Aktif grup bulunamadı.")
            return

        success_count = 0
        fail_count = 0

        for chat_id in list(known_group_ids):
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=text_to_send
                )
                success_count += 1
            except Exception as e:
                logger.error("Grup %s hatası: %s", chat_id, e)
                fail_count += 1

        await message.reply_text(
            f"✅ Tamamlandı!\n"
            f"🟢 Başarılı: {success_count}\n"
            f"🔴 Başarısız: {fail_count}"
        )
        return

    if not text or text.startswith("/"):
        return

    bot_username = context.bot.username
    is_mentioned = False

    if bot_username and f"@{bot_username}" in text:
        is_mentioned = True
        text = text.replace(f"@{bot_username}", "").strip()
    elif message.reply_to_message and message.reply_to_message.from_user.id == context.bot.id:
        is_mentioned = True

    # 1. BOT ETİKETLENDİYSE -> ZEKİ ASİSTAN MODU
    if is_mentioned:
        if not text:
            text = "Nasıl yardımcı olabilirim?"

        logger.info("AI Assistant Request: %s", text)

        ai_response = ask_openai(ASSISTANT_PROMPT, text)

        if ai_response:
            await message.reply_text(ai_response)
        return

    # 2. BOT ETİKETLENMEDİYSE -> C2 PROFESYONEL ÇEVİRİ (TÜRKÇE - RUSÇA)
    logger.info("Translation request: %s", text)

    translation = ask_openai(TRANSLATION_PROMPT, text)

    if translation is None or not translation:
        return

    await message.reply_text(
        translation,
        disable_web_page_preview=True,
    )


# =========================================================
# ERROR HANDLER
# =========================================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Telegram Error: %s", context.error, exc_info=True)


# =========================================================
# MAIN
# =========================================================

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN bulunamadı!")

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY bulunamadı!")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("hakkinda", about_handler))
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_error_handler(error_handler)

    logger.info("🤖 Viyana AI zeki asistan ve profesyonel çeviri botu aktif!")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
