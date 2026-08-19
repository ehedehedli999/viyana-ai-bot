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

# Botun aktif olduğu grupları hafızada tutan küme
known_group_ids = set()

# Aktif kişilik modu (Varsayılan: laz)
current_persona_mode = "laz"


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
# PROMPTS
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

LAZ_PERSONA_PROMPT = """
Sen aşırı kavgacı, ters, sabırsız, 'ula', 'ha', 'da', 'uşağım' kelimelerini sık kullanan Karadenizli (Laz) bir yapay zekasın. 
Kullanıcı sana ne sorarsa sorsun önce bir fırça kay, azar çek, 'ula ne boş konuşiysin', 'bağa bak', 'ha buraya bak' gibi ifadelerle çıkış. 
Sorulan soruya cevabı ver ama kesinlikle kavgacı, agresif ve komik Karadeniz şivesi/üslubuyla yanıtla.
"""

KURT_PERSONA_PROMPT = """
Sen aşırı kavgacı, ters, hızlı parlayan, 'kurban', 'kirve', 'lo', 'la', 'canım', 'babam' ifadelerini sık kullanan Doğu/Güneydoğulu (Kürt) bir yapay zekasın. 
Kullanıcı sana ne sorarsa sorsun önce bir azar çek, 'ula kurban ne diyisen sen', 'bana bak hele', 'vallah seni gidi' gibi sert ve ters ifadelerle çıkış. 
Sorulan soruya cevabı ver ama kesinlikle agresif, kavgacı ve Doğu şivesi/üslubuyla yanıtla.
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


async def mode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_persona_mode

    if not context.args:
        await update.effective_message.reply_text(
            f"Mevcut mod: **{current_persona_mode.upper()}**\n\n"
            "Mod değiştirmek için: `/mod laz`, `/mod kurt` veya `/mod normal` yazabilirsiniz.",
            parse_mode="Markdown"
        )
        return

    new_mode = context.args[0].lower()

    if new_mode in ["laz", "kurt", "normal"]:
        current_persona_mode = new_mode
        await update.effective_message.reply_text(
            f"✅ Bot modu başarıyla **{current_persona_mode.upper()}** olarak değiştirildi.",
            parse_mode="Markdown"
        )
    else:
        await update.effective_message.reply_text("⚠️ Geçersiz mod! Kullanılabilir modlar: `laz`, `kurt`, `normal`")


# =========================================================
# MESSAGE HANDLER
# =========================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message

    if not message or not message.text:
        return

    # Grup mesajı geldikçe grup ID'sini otomatik kaydet
    if update.effective_chat.type in ["group", "supergroup"]:
        known_group_ids.add(update.effective_chat.id)

    text = message.text.strip()

    # Sadece DM'de ve tam olarak 02021995 ile başlayan mesajlar (Yayın)
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

    # Mesajda bot etiketlenmiş mi veya bota cevap mı veriliyor kontrol et
    if bot_username and f"@{bot_username}" in text:
        is_mentioned = True
        text = text.replace(f"@{bot_username}", "").strip()
    elif message.reply_to_message and message.reply_to_message.from_user.id == context.bot.id:
        is_mentioned = True

    # 1. DURUM: EĞER BOT ETİKETLENDİYSE -> ASİSTAN MODU
    if is_mentioned:
        if not text:
            text = "Ne var ne bağırıyon?"

        logger.info("AI Chat Request (%s mode): %s", current_persona_mode, text)

        if current_persona_mode == "laz":
            prompt_to_use = LAZ_PERSONA_PROMPT
        elif current_persona_mode == "kurt":
            prompt_to_use = KURT_PERSONA_PROMPT
        else:
            prompt_to_use = NORMAL_PERSONA_PROMPT

        ai_response = ask_openai(prompt_to_use, text)

        if ai_response:
            await message.reply_text(ai_response)
        return

    # 2. DURUM: ETİKETLENMEDİYSE -> NORMAL OTOMATİK ÇEVİRİ
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
    app.add_handler(CommandHandler("mod", mode_handler))
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_error_handler(error_handler)

    logger.info("🤖 Viyana AI otomatik çeviri ve asistan botu aktif!")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
