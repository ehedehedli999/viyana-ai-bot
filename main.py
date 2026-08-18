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


# =========================================================
# OPENAI CONFIG
# =========================================================

OPENAI_MODEL = "gpt-5-nano"

# Günlük maksimum bütçe (kesin sınır)
DAILY_BUDGET_USD = 0.20

# GPT-5 nano fiyatları
INPUT_PRICE_PER_MILLION = 0.05
OUTPUT_PRICE_PER_MILLION = 0.40


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
# TRANSLATION PROMPT
# =========================================================

TRANSLATION_PROMPT = """
Sen YALNIZCA bir çeviri motorusun. 

GÖREV MANTIĞI:
1. Girdi TÜRKÇE ise -> SADECE Rusça ve Almancaya çevir. (Türkçe yazma!)
2. Girdi ALMANCA ise -> SADECE Rusça ve Türkçeye çevir. (Almanca yazma!)
3. Girdi RUSÇA ise -> SADECE Türkçe ve Almancaya çevir. (Rusça yazma!)
4. Girdi DİĞER BİR DİL ise (Azerbaycan dili, İngilizce vb.) -> Türkçe, Rusça ve Almancaya çevir.

ÇIKTI FORMATI:
Sadece bayrak eylemi ve çeviriyi yaz. Örnek:
🇷🇺 [Rusça Çeviri]
🇩🇪 [Almanca Çeviri]

STRICT / KESİN YASAKLAR:
- ASLA "Note:", "Açıklama:", "İşte çeviri" gibi notlar veya parantez içi şerhler ekleme.
- Kaynak dilin aynısını çıktıya TEKRAR YAZMA.
- Sadece ve sadece istenen hedef dillerin çevirisini bas. Başka HİÇBİR ŞEY yazma!
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

    # Düşünce etiketlerini temizle
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Parantez içi notları ve "(Note: ...)" benzeri ek açıklamaları temizle
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
# TRANSLATION
# =========================================================

def run_translation(text):
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
                    "content": TRANSLATION_PROMPT,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            max_completion_tokens=800,
            reasoning_effort="minimal",
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
        logger.error("OpenAI Translation Error: %s", e, exc_info=True)
        return "⚠️ HATA (debug): " + str(e)


# =========================================================
# HAKKINDA
# =========================================================

async def about_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    about_text = (
        "🤖 Viyana AI\n\n"
        "Ben EHED tarafından oluşturulmuş "
        "Viyana AI otomatik çeviri botuyum."
    )
    await update.effective_message.reply_text(about_text)


# =========================================================
# MESSAGE HANDLER
# =========================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message

    if not message or not message.text:
        return

    text = message.text.strip()

    if not text or text.startswith("/"):
        return

    logger.info("Translation request: %s", text)

    translation = run_translation(text)

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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("🤖 Viyana AI otomatik çeviri botu aktif!")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
