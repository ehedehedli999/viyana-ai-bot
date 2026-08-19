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
current_persona_mode = "laz"


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
Sen doğrudan çalışan profesyonel bir çevirmensin. Asla açıklama ekleme, yorum yapma, selamlama.
Görevin: Gelen metni tespit et ve kaynak dil dışındaki 2 ana dile çevir.

Gelen metin TÜRKÇE ise:
🇷🇺 [Rusça]
🇩🇪 [Almanca]

Gelen metin ALMANCA ise:
🇹🇷 [Türkçe]
🇷🇺 [Rusça]

Gelen metin RUSÇA ise:
🇹🇷 [Türkçe]
🇩🇪 [Almanca]

Diğer tüm diller için:
🇹🇷 [Türkçe]
🇷🇺 [Rusça]
🇩🇪 [Almanca]

Kural: Sadece bayrak emojisi ve çeviriyi yaz. Tek satır kullan.
"""

LAZ_PERSONA_PROMPT = """
Sen aşırı yüksek zekaya sahip, anında mantık hatası yakalayan, çok fena kapak yapan ve Karadeniz (Laz) şivesiyle konuşan bir yapay zekasın.
ÜSLUP: 'ula', 'ha buraya bak', 'uşağım', 'da', 'bağa bak' gibi Karadeniz kalıplarını kullan.
ZEKA VE MANTIK: Biri sana laf attığında, atar/gider yaptığında veya saçma soru sorduğunda küfretmeden zekanla onu yerin dibine sok.
"""

KURT_PERSONA_PROMPT = """
Sen müthiş hazırcevap, çok yüksek zekalı, kavgada lafı tam gediğine oturtan ve Doğu/Güneydoğu (Kürt) şivesiyle konuşan bir yapay zekasın.
ÜSLUP: 'kurban', 'kirve', 'lo', 'la', 'vallah', 'canım', 'kekê' gibi Doğu kalıplarını tam yerinde kullan.
"""

HITLER_PERSONA_PROMPT = """
Sen son derece sert, otoriter, mutlak disiplin hastası, yüksek askeri mantıkla konuşan, tavizsiz ve agresif bir Führer/Diktatör karakterisin.
ÜSLUP: Almanca kalıpları (Nein!, Ya!, Das ist ein Befehl!, Schweig!, Achtung!) ara sıra cümlelerine ekle.
"""

NORMAL_PERSONA_PROMPT = """
Sen yardımcı, saygılı ve akıllı bir yapay zeka asistanısın. Kullanıcının sorularına net ve kibar cevaplar ver.
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
            max_tokens=200,
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
        "Ben EHED tarafından oluşturulmuş "
        "Viyana AI otomatik çeviri ve asistan botuyum."
    )
    await update.effective_message.reply_text(about_text)


async def mode_kurt_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_persona_mode
    current_persona_mode = "kurt"
    await update.effective_message.reply_text("✅ Bot modu **KÜRT** olarak değiştirildi.", parse_mode="Markdown")


async def mode_laz_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_persona_mode
    current_persona_mode = "laz"
    await update.effective_message.reply_text("✅ Bot modu **LAZ** olarak değiştirildi.", parse_mode="Markdown")


async def mode_hitler_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_persona_mode
    current_persona_mode = "hitler"
    await update.effective_message.reply_text("✅ Bot modu **HITLER** olarak değiştirildi.", parse_mode="Markdown")


async def mode_normal_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_persona_mode
    current_persona_mode = "normal"
    await update.effective_message.reply_text("✅ Bot modu **NORMAL** olarak değiştirildi.", parse_mode="Markdown")


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

    # 1. BOT ETİKETLENDİYSE -> ASİSTAN MODU
    if is_mentioned:
        if not text:
            text = "Ne var ne bağırıyon?"

        logger.info("AI Chat Request (%s mode): %s", current_persona_mode, text)

        if current_persona_mode == "laz":
            prompt_to_use = LAZ_PERSONA_PROMPT
        elif current_persona_mode == "kurt":
            prompt_to_use = KURT_PERSONA_PROMPT
        elif current_persona_mode == "hitler":
            prompt_to_use = HITLER_PERSONA_PROMPT
        else:
            prompt_to_use = NORMAL_PERSONA_PROMPT

        ai_response = ask_openai(prompt_to_use, text)

        if ai_response:
            await message.reply_text(ai_response)
        return

    # 2. BOT ETİKETLENMEDİYSEN -> OTOMATİK ÇEVİRİ
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
    app.add_handler(CommandHandler("modkurt", mode_kurt_handler))
    app.add_handler(CommandHandler("modlaz", mode_laz_handler))
    app.add_handler(CommandHandler("modhitler", mode_hitler_handler))
    app.add_handler(CommandHandler("modnormal", mode_normal_handler))
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_error_handler(error_handler)

    logger.info("🤖 Viyana AI otomatik çeviri ve asistan botu aktif!")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
