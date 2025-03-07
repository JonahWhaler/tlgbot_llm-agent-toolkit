"""
Entrypoint
"""

import logging
import time
from typing import Any, Callable, Coroutine

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
from pydantic import BaseModel
from mystorage import SQLite3_Storage
import handlers
import sa_handlers
import myconfig


class TimeoutModel(BaseModel):
    connect: int
    pool: int
    read: int
    write: int
    media_write: int

    def __mul__(self, factor: float) -> "TimeoutModel":
        return TimeoutModel(
            connect=self.connect / factor,
            pool=self.pool / factor,
            read=self.read / factor,
            write=self.write / factor,
            media_write=self.media_write / factor,
        )


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
    bot.add_handler(CommandHandler("usage", handlers.show_usage), group=1)
    # bot.add_handler(
    #     CommandHandler("compress", handlers.compress_memory_handler), group=1
    # )
    bot.add_handler(
        CommandHandler("mycharacter", handlers.show_character_handler), group=1
    )
    bot.add_handler(
        CommandHandler("chat_models", handlers.show_chat_model_menu), group=1
    )
    bot.add_handler(
        CommandHandler("vision_models", handlers.show_vision_model_menu), group=1
    )
    bot.add_handler(
        CommandHandler("system_chat_models", sa_handlers.show_chat_model_menu),
        group=1,
    )
    bot.add_handler(
        CommandHandler("system_vision_models", sa_handlers.show_vision_model_menu),
        group=1,
    )
    bot.add_handler(
        CommandHandler("pending", sa_handlers.pending_user_handler), group=1
    )
    bot.add_handler(
        CommandHandler("allow", sa_handlers.allow_pending_user_handler), group=1
    )

    # Buttons
    bot.add_handler(
        telegram.ext.CallbackQueryHandler(
            handlers.set_chat_model_handler, pattern=r"^set_chat_model"
        ),
        group=1,
    )
    bot.add_handler(
        telegram.ext.CallbackQueryHandler(
            handlers.set_vision_model_handler, pattern=r"^set_vision_model"
        ),
        group=1,
    )
    bot.add_handler(
        telegram.ext.CallbackQueryHandler(
            sa_handlers.set_chat_model_handler, pattern=r"^set_sys_chat_model"
        ),
        group=1,
    )
    bot.add_handler(
        telegram.ext.CallbackQueryHandler(
            sa_handlers.set_vision_model_handler, pattern=r"^set_sys_vision_model"
        ),
        group=1,
    )

    bot.add_handler(MessageHandler(filters.TEXT, handlers.message_handler), group=1)
    bot.add_handler(MessageHandler(filters.PHOTO, handlers.photo_handler), group=1)
    bot.add_handler(MessageHandler(filters.VOICE, handlers.voice_handler), group=1)
    bot.add_handler(MessageHandler(filters.AUDIO, handlers.audio_handler), group=1)
    bot.add_error_handler(handlers.error_handler)
    bot.run_polling(poll_interval=1.0)


def build(
    tk: str,
    config: TimeoutModel,
    rate_limiter: AIORateLimiter,
    post_init_callback: Callable[[Application], Coroutine[Any, Any, None]],
) -> Application:
    return (
        ApplicationBuilder()
        .token(tk)
        .concurrent_updates(True)
        .connect_timeout(config.connect)
        .read_timeout(config.read)
        .write_timeout(config.write)
        .media_write_timeout(config.media_write)
        .pool_timeout(config.pool)
        .rate_limiter(rate_limiter)
        .post_init(post_init_callback)
        .build()
    )


def update_tf(tf: float, mf: float = 1.2, _max: float = 10) -> float:
    return round(min(tf * mf, _max), 1)


def update_df(d: float, mf: float = 1.2, _max: float = 60.0) -> float:
    """Update delay time. (seconds)"""
    return round(min(d * mf, _max), 1)


async def post_init(app: Application) -> None:
    cmds = [
        ("/start", "Start Message"),
        # ("/compress", "Compress Memory"),
        ("/new", "Start a new chat"),
        ("/clear", "Clear Memory"),
        ("/mycharacter", "Show My Character"),
        ("/usage", "Show My Usage"),
        # ("/characters", "Show Character Menu"),
        ("/chat_models", "Show Chat Completion Models"),
        ("/vision_models", "Show Vision Interpretation Models"),
        ("/help", "Help Message"),
    ]
    await app.bot.set_my_commands(cmds)


def main(resource_dict: dict):
    logger = logging.getLogger(__name__)

    timeout_object = TimeoutModel(
        connect=5.0, pool=1.0, read=5.0, write=5.0, media_write=20.0
    )
    timeout_factor = 1.2
    delay, delay_factor = 5.0, 1.5

    token = myconfig.TLG_TOKEN
    assert isinstance(token, str)
    max_retry: str | int | None = myconfig.TLG_MAX_RETRY
    if max_retry is None:
        max_retry = 5
    elif isinstance(max_retry, str):
        max_retry = int(max_retry)

    while True:
        try:
            aio_rate_limiter = AIORateLimiter(
                overall_max_rate=10, overall_time_period=1, max_retries=max_retry
            )
            application = build(token, timeout_object, aio_rate_limiter, post_init)
            application.bot_data["secret_counter"] = 987
            # for key, resource in resource_dict.items():
            #     application.bot_data[key] = resource

            run_bot(application)
            break
        except telegram.error.TimedOut as error:
            logger.error(
                "ErrorType: %s, ErrorMessage: %s", type(error), str(error)
            )  # AttributeError: type object 'TimedOut' has no attribute 'name'
            # Update timeout
            timeout_factor = update_tf(timeout_factor)
            timeout_object = timeout_object * timeout_factor
            # Update delay
            delay = update_df(delay, delay_factor)
            time.sleep(delay)


def init():
    logging.basicConfig(
        filename="/log/tlg_bot.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARN)

    sys_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "system", False)
    sys_sql3_table.set(
        "chat-completion", {"provider": "gemini", "model_name": "gemini-1.5-flash"}
    )
    sys_sql3_table.set(
        "image-interpretation", {"provider": "ollama", "model_name": "llava:7b"}
    )
    sys_sql3_table.set("transcription", {"provider": "local", "model_name": "turbo"})


if __name__ == "__main__":
    init()
    global_dict = {}
    main(global_dict)
