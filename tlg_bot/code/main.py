"""
Entrypoint
"""

import logging
import time
from typing import Any, Callable, Coroutine, Tuple

import telegram
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    AIORateLimiter,
    filters,
)
import telegram.ext
import handlers
import config

logging.basicConfig(
    filename="/log/tlg_bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARN)


def run_bot(bot: Application) -> None:
    bot.add_handler(
        MessageHandler(filters.ALL, handlers.middleware_function),
        group=0,
    )
    # bot.add_handler(CommandHandler("export", handlers.export_handler, filters=allowed_list), group=1)
    bot.add_handler(CommandHandler("help", handlers.help_handler), group=1)
    bot.add_handler(CommandHandler("start", handlers.start_handler), group=1)
    bot.add_handler(CommandHandler("new", handlers.reset_chatmemory_handler), group=1)
    bot.add_handler(CommandHandler("clear", handlers.reset_user_handler), group=1)
    bot.add_handler(
        CommandHandler("compress", handlers.compress_memory_handler), group=1
    )
    bot.add_handler(
        CommandHandler("mycharacter", handlers.show_character_handler), group=1
    )
    bot.add_handler(CommandHandler("characters", handlers.show_character_menu), group=1)
    bot.add_handler(CommandHandler("models", handlers.show_model_menu), group=1)
    bot.add_handler(
        CommandHandler("creativity", handlers.show_creativity_menu), group=1
    )
    # Buttons
    bot.add_handler(
        telegram.ext.CallbackQueryHandler(
            handlers.set_model_handler, pattern=r"^set_model"
        )
    )
    bot.add_handler(
        telegram.ext.CallbackQueryHandler(
            handlers.set_character_handler, pattern=r"^set_character"
        )
    )
    bot.add_handler(
        telegram.ext.CallbackQueryHandler(
            handlers.set_creativity_handler, pattern=r"^set_creativity"
        )
    )

    bot.add_handler(MessageHandler(filters.TEXT, handlers.message_handler), group=1)
    bot.add_handler(MessageHandler(filters.PHOTO, handlers.photo_handler), group=1)
    bot.add_handler(MessageHandler(filters.VOICE, handlers.voice_handler), group=1)
    bot.add_error_handler(handlers.error_handler)
    bot.run_polling(poll_interval=1.0)


def build(
    tk: str,
    cto: float,
    rto: float,
    wto: float,
    media_wto: float,
    pto: float,
    rate_limiter: AIORateLimiter,
    post_init_callback: Callable[[Application], Coroutine[Any, Any, None]],
) -> Application:
    return (
        ApplicationBuilder()
        .token(tk)
        .concurrent_updates(True)
        .connect_timeout(cto)
        .read_timeout(rto)
        .write_timeout(wto)
        .media_write_timeout(media_wto)
        .pool_timeout(pto)
        .rate_limiter(rate_limiter)
        .post_init(post_init_callback)
        .build()
    )


def update_timeout_factor(tf: float, mf: float = 1.2, _max: float = 10) -> float:
    return round(min(tf * mf, _max), 1)


def update_delay(d: float, mf: float = 1.2, _max: float = 60.0) -> float:
    """Update delay time. (seconds)"""
    return round(min(d * mf, _max), 1)


async def post_init(app: Application) -> None:
    cmds = [
        ("/start", "Start Message"),
        # ("/compress", "Compress Memory"),
        ("/new", "Start a new chat"),
        ("/clear", "Clear Memory"),
        ("/mycharacter", "Show My Character"),
        # ("/characters", "Show Character Menu"),
        ("/models", "Show Model Menu"),
        ("/help", "Help Message"),
    ]
    await app.bot.set_my_commands(cmds)


def update_timeout(
    factor: float,
    _connect_timeout: float,
    _read_timeout: float,
    _write_timeout: float,
    _media_write_timeout: float,
    _pool_timeout: float,
) -> Tuple[float, float, float, float, float]:
    return (
        _connect_timeout * factor,
        _read_timeout * factor,
        _write_timeout * factor,
        _media_write_timeout * factor,
        _pool_timeout * factor,
    )


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    connect_timeout, pool_timeout = 5.0, 1.0
    read_timeout, write_timeout, media_write_timeout = 5.0, 5.0, 20.0
    timeout_factor = 1.2
    delay, delay_factor = 5.0, 1.5

    token = config.TLG_TOKEN
    assert isinstance(token, str)
    max_retry: str | int | None = config.TLG_MAX_RETRY
    if max_retry is None:
        max_retry = 5
    elif isinstance(max_retry, str):
        max_retry = int(max_retry)

    while True:
        try:
            aio_rate_limiter = AIORateLimiter(
                overall_max_rate=10, overall_time_period=1, max_retries=max_retry
            )
            application = build(
                token,
                connect_timeout,
                read_timeout,
                write_timeout,
                media_write_timeout,
                pool_timeout,
                aio_rate_limiter,
                post_init,
            )
            run_bot(application)
            break
        except telegram.error.TimedOut as error:
            logger.error(
                "ErrorType: %s, ErrorMessage: %s", type(error), str(error)
            )  # AttributeError: type object 'TimedOut' has no attribute 'name'
            # Update timeout
            timeout_factor = update_timeout_factor(timeout_factor)
            (
                connect_timeout,
                read_timeout,
                write_timeout,
                media_write_timeout,
                pool_timeout,
            ) = update_timeout(
                timeout_factor,
                connect_timeout,
                read_timeout,
                write_timeout,
                media_write_timeout,
                pool_timeout,
            )
            # Update delay
            delay = update_delay(delay, delay_factor)
            time.sleep(delay)
