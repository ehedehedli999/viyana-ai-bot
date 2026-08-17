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
# Render Dashboard'a yazılacak:
#
# TELEGRAM_BOT_TOKEN
# OPENAI_API_KEY
# =========================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =========================================================
# OPENAI CONFIG
# =========================================================

# Çok düşük maliyetli model.
OPENAI_MODEL = "gpt-5-nano"

# Günlük güvenlik bütçesi
DAILY_BUDGET_USD = 0.20

# Güvenlik payı bırakıyoruz.
# Kod yaklaşık $0.18'e ulaştığında yeni istekleri durdurur.
DAILY_SOFT_LIMIT_USD = 0.18

# gpt-5-nano fiyatları:
# Input:  $0.05 / 1M tokens
# Output: $0.40 / 1M tokens

INPUT_PRICE_PER_MILLION = 0.05
OUTPUT_PRICE_PER_MILLION = 0.40


# =========================================================
# GÜNLÜK KULLANIM TAKİBİ
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


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (
        input_tokens / 1_000_000
    ) * INPUT_PRICE_PER_MILLION

    output_cost = (
        output_tokens / 1_000_000
    ) * OUTPUT_PRICE_PER_MILLION

    return input_cost + output_cost


def current_daily_cost() -> float:
    return calculate_cost(
        daily_input_tokens,
        daily_output_tokens,
    )


# =========================================================
# TRANSLATION PROMPT
# =========================================================

TRANSLATION_PROMPT = """
Sen profesyonel bir simultane çeviri motorusun.

GÖREV:

1. Mesaj Türkçeyse SADECE Rusçaya çevir.
2. Mesaj Rusçaysa SADECE Türkçeye çevir.
3. Mesaj Azerbaycancaysa hem Türkçeye hem Rusçaya çevir.
4. Mesaj İngilizce, Almanca veya başka bir dildeyse hem Türkçeye hem Rusçaya çevir.

ÇEVİRİ KALİTESİ:

- Ana dili konuşan insan seviyesinde, C2 düzeyinde çeviri yap.
- Anlamı eksiksiz koru.
- Kelime kelime mekanik çeviri yapma; hedef dilde doğal ve doğru ifade kullan.
- Ancak kaynak metinde olmayan hiçbir anlam ekleme.
- Yorum yapma.
- Açıklama yapma.
- Özetleme yapma.
- Cevap verme.
- Selamlaşma ekleme.
- "İşte çeviri:", "Tabii:", "Elbette:" gibi ifadeler ekleme.
- Kullanıcının cümlesini değiştirme veya güzelleştirme.
- Argo, günlük konuşma, küfür veya kaba ifadeler varsa anlamını koru.
- Özel isimleri mümkün olduğunca aynen koru.
- Sayıları, tarihleri ve önemli bilgileri değiştirme.
- Emojileri mümkün olduğunca koru.
- Mesaj çok kısa olsa bile sadece çevirisini ver.

ÖRNEK:

Türkçe:
Merhaba

ÇIKTI:
🇷🇺 Привет

Türkçe:
Tamam

ÇIKTI:
🇷🇺 Хорошо

Rusça:
Привет

ÇIKTI:
🇹🇷 Merhaba

Rusça:
Хорошо

ÇIKTI:
🇹🇷 Tamam

Azerbaycanca:
Salam, necəsən?

ÇIKTI:
🇹🇷 Merhaba, nasılsın?
🇷🇺 Привет, как ты?

KESİN KURAL:

Çıktıda yalnızca çeviri bulunmalıdır.
Başka hiçbir açıklama veya cümle yazma.
"""


# =========================================================
# OPENAI CLIENT
# =========================================================

def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)


# =========================================================
# RESPONSE TEMİZLEME
# =========================================================

def clean_response(text: str) -> str:
    if not text:
        return ""

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return text.strip()


# =========================================================
# TRANSLATION
# =========================================================

def run_translation(text: str) -> str:
    global daily_input_tokens
    global daily_output_tokens

    reset_daily_usage_if_needed()

    # Günlük limite ulaşıldıysa API çağrısı yapma.
    if current_daily_cost() >= DAILY_SOFT_LIMIT_USD:
        logger.warning(
            "Günlük API bütçe sınırına ulaşıldı: $%.6f",
            current_daily_cost(),
        )
        return ""

    try:
        client = get_openai_client()

        response = client.chat.completions.create(
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
            temperature=0,
            max_tokens=300,
        )

        # Token kullanımını al
        usage = getattr(response, "usage", None)

        if usage:
            input_tokens = getattr(
                usage,
                "prompt_tokens",
                0,
            )

            output_tokens = getattr(
                usage,
                "completion_tokens",
                0,
            )

            daily_input_tokens += input_tokens
            daily_output_tokens += output_tokens

            logger.info(
                "Translation usage: input=%s output=%s daily_cost=$%.6f",
                input_tokens,
                output_tokens,
                current_daily_cost(),
            )

        output = response.choices[0].message.content

        return clean_response(output)

    except Exception as e:
        logger.error(
            "Translation Error: %s",
            e,
            exc_info=True,
        )
        return ""


# =========================================================
# HAKKINDA
# =========================================================

async def about_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    about_text = (
        "🤖 Viyana AI\n\n"
        "Ben EHET tarafından oluşturulmuş "
        "Viyana AI otomatik çeviri botuyum."
    )

    await update.effective_message.reply_text(
        about_text
    )


# =========================================================
# MESAJ HANDLER
# =========================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.effective_message

    if not message or not message.text:
        return

    text = message.text.strip()

    if not text:
        return

    # Komutlar burada işlenmez.
    if text.startswith("/"):
        return

    logger.info(
        "Translation request: %s",
        text,
    )

    translation = run_translation(text)

    if translation:
        await message.reply_text(
            translation,
            disable_web_page_preview=True,
        )


# =========================================================
# ERROR HANDLER
# =========================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):
    logger.error(
        "Telegram Error: %s",
        context.error,
        exc_info=True,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN bulunamadı!"
        )

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY bulunamadı!"
        )

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    # SADECE hakkında komutu
    app.add_handler(
        CommandHandler(
            "hakkinda",
            about_handler,
        )
    )

    # Otomatik çeviri
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    app.add_error_handler(
        error_handler
    )

    logger.info(
        "🤖 Viyana AI otomatik çeviri botu aktif!"
    )

    app.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
