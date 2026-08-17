import logging
import os
import re

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
# LOGGING SETUP
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# =========================================================
# GEMINI ENGINE CONFIG
# =========================================================

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


# =========================================================
# SYSTEM PROMPTS
# =========================================================

TRANSLATION_PROMPT = """
Sen ana dili düzeyinde (C2) anında çeviri yapan profesyonel bir yapay zekasın.

GÖREV VE DİL ALGILAMA KURALLARI:
1. Gelen mesaj Türkçeyse: SADECE Rusçaya çevir.
2. Gelen mesaj Rusçaysa: SADECE Türkçeye çevir.
3. Gelen mesaj Azerbaycanca, İngilizce, Almanca veya başka bir dildeyse: Hem Türkçeye hem de Rusçaya çevir.

KALİTE VE FORMAT KURALLARI:
- Çeviriyi C2 seviyesinde, kelimesi kelimesine doğru, doğal ve akıcı yap.
- "hım", "evet", "tamam", "oldu" gibi çok kısa ifadeleri bile eksiksiz ve doğru karşılığıyla çevir.
- Uydurma kelime, yorum, açıklama, selamlaşma veya sohbet cümlesi KESİNLİKLE EKLEME.
- Düşünme adımlarını veya hafıza loglarını KESİNLİKLE ÇIKTIYA YAZMA.
- Sadece ve sadece aşağıdaki bayraklı çıktı formatını kullan:

FORMATLAR:
Kaynak Türkçeyse:
🇷🇺 Rusça: [çeviri]

Kaynak Rusçaysa:
🇹🇷 Türkçe: [çeviri]

Kaynak Diğer Dillerse:
🇹🇷 Türkçe: [çeviri]
🇷🇺 Rusça: [çeviri]
"""

CHAT_PROMPT = """
Sen Viyana AI altyapısıyla çalışan akıllı, yardımsever ve doğal bir yapay zeka asistanısın.

KURALLAR:
1. Kullanıcı sana hangi dilde yazdıysa (Türkçe, Rusça, Azerbaycanca vs.) kesinlikle O DİLDE cevap ver.
2. Cevapların net, doğru ve arkadaş canlısı olsun.
3. Gereksiz düşünme adımlarını veya iç logları çıktıya yansıtma.
"""


# =========================================================
# HELPER FUNCTIONS
# =========================================================


def get_gemini_client():
    return OpenAI(
        base_url=GEMINI_BASE_URL,
        api_key=GEMINI_API_KEY,
    )


def clean_response(text: str) -> str:
    if not text:
        return ""
    text = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    return text.strip()


def run_translation(text: str) -> str:
    client = get_gemini_client()
    try:
        response = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": TRANSLATION_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        output = response.choices[0].message.content
        return clean_response(output)
    except Exception as e:
        logger.error("Translation Error: %s", e)
        return ""


def run_ai_chat(text: str) -> str:
    client = get_gemini_client()
    try:
        response = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": CHAT_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.7,
            max_tokens=600,
        )
        output = response.choices[0].message.content
        return clean_response(output)
    except Exception as e:
        logger.error("Chat Error: %s", e)
        return "⚠️ Üzgünüm, şu an yanıt veremiyorum."


# =========================================================
# HANDLERS
# =========================================================


async def about_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    about_text = (
        "🤖 **Viyana AI**\n\n"
        "Ben Ehed Tarafından Oluşturulan Viyana AI Asistan Ve Otomatik Çeviri Botuyum."
    )
    await update.effective_message.reply_text(
        about_text, parse_mode="Markdown"
    )


async def handle_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    message = update.effective_message
    if not message or not message.text:
        return

    text = message.text.strip()

    # Hakkında Butonu / Metni Kontrolü
    if text.lower() in [
        "haqqında",
        "haqqinda",
        "about",
        "hakkında",
        "hakkinda",
    ]:
        await about_handler(update, context)
        return

    if text.startswith("/"):
        return

    bot_username = context.bot.username
    is_tagged = False

    # Etiketlenme veya Mesaja Yanıt Kontrolü
    if bot_username and f"@{bot_username}" in text:
        is_tagged = True
    elif (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == context.bot.id
    ):
        is_tagged = True

    # 1. MOD: Etiketlendiğinde (Yapay Zeka Sohbet Modu)
    if is_tagged:
        clean_prompt = (
            text.replace(f"@{bot_username}", "").strip()
            if bot_username
            else text
        )
        if not clean_prompt:
            clean_prompt = "Merhaba"

        logger.info("Chat mode triggered: %s", clean_prompt)
        reply = run_ai_chat(clean_prompt)
        await message.reply_text(reply)

    # 2. MOD: Etiketlenmediğinde (Otomatik Grup Çevirisi)
    else:
        logger.info("Translation mode triggered: %s", text)
        translation = run_translation(text)
        if translation:
            await message.reply_text(
                translation, disable_web_page_preview=True
            )


async def error_handler(
    update: object, context: ContextTypes.DEFAULT_TYPE
):
    logger.error("Telegram Error: %s", context.error)


# =========================================================
# MAIN EXECUTION
# =========================================================


def main():
    if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN ve GEMINI_API_KEY bulunamadı!"
        )

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Komutlar
    app.add_handler(
        CommandHandler(
            ["about", "haqqinda", "hakkinda"], about_handler
        )
    )
    app.add_handler(CommandHandler("start", about_handler))

    # Mesaj Dinleyici
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )
    app.add_error_handler(error_handler)

    logger.info("🤖 Bot aktif ve dinlemede!")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

