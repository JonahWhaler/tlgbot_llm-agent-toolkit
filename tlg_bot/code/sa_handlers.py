import logging
from typing import Optional
from asyncio import Lock

# External Imports
import telegram
from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode

import myconfig
from mystorage import SQLite3_Storage
from custom_library import format_identifier

sa_locks: dict[str, Lock] = {}
# Please implement cleanup mechanism when the number of the system admin
# is expected to scale up.


def get_sa_lock(identifier: str) -> Lock:
    """
    Get a lock for a specific user based on their identifier.

    Args:
        identifier (str): The identifier of the user.

    Returns:
        asyncio.Lock: The lock associated with the user.
    """
    global sa_locks
    if identifier not in sa_locks:
        sa_locks[identifier] = Lock()
    return sa_locks[identifier]


async def show_chat_model_menu(update: Update, context: CallbackContext) -> None:
    logger = logging.getLogger(__name__)

    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_sa_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        output_string = "Click to select a chat model:\n"

        keyboard = []
        providers = list(myconfig.PROVIDER.keys())
        for provider in providers:
            models = list(myconfig.PROVIDER[provider]["t2t"])
            for model_name in models:
                name = f"{provider} - {model_name}"
                keyboard.append(
                    [
                        telegram.InlineKeyboardButton(
                            name,
                            callback_data=f"set_sys_chat_model|{provider}$$${model_name}",
                        )
                    ]
                )

        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        await message.reply_text(
            output_string, reply_markup=reply_markup, parse_mode=ParseMode.HTML
        )


async def show_vision_model_menu(update: Update, context: CallbackContext) -> None:
    logger = logging.getLogger(__name__)
    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_sa_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)

        output_string = "Click to select a vision model:\n"

        keyboard = []
        providers = list(myconfig.PROVIDER.keys())
        for provider in providers:
            if "i2t" not in myconfig.PROVIDER[provider]:
                continue

            models = list(myconfig.PROVIDER[provider]["i2t"])
            for model_name in models:
                name = f"{provider} - {model_name}"
                keyboard.append(
                    [
                        telegram.InlineKeyboardButton(
                            name,
                            callback_data=f"set_sys_vision_model|{provider}$$${model_name}",
                        )
                    ]
                )

        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        await message.reply_text(
            output_string, reply_markup=reply_markup, parse_mode=ParseMode.HTML
        )


async def set_chat_model_handler(update: Update, context: CallbackContext) -> None:
    logger = logging.getLogger(__name__)

    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    callback_query = update.callback_query

    if callback_query is None:
        raise ValueError("Callback query is None.")

    message = callback_query.message

    if message is None:
        raise ValueError("Message is None.")

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_sa_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        query = update.callback_query
        await query.answer()

        provider_model = query.data.split("|")[1]
        provider, model_name = provider_model.split("$$$")
        provider_model = query.data.split("|")[1]
        provider, model_name = provider_model.split("$$$")
        sys_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "system", False)
        sys_sql3_table.set(
            "chat-completion", {"provider": provider, "model": model_name}
        )

        await context.bot.send_message(
            chat_id=message.chat.id,
            text=f"Chat Completion model set to {provider} - {model_name}",
            parse_mode=ParseMode.HTML,
        )

    logger.info("Released lock for user: %s", identifier)


async def set_vision_model_handler(update: Update, context: CallbackContext) -> None:
    logger = logging.getLogger(__name__)

    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    callback_query = update.callback_query

    if callback_query is None:
        raise ValueError("Callback query is None.")

    message = callback_query.message

    if message is None:
        raise ValueError("Message is None.")

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_sa_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)

        query = update.callback_query
        await query.answer()

        provider_model = query.data.split("|")[1]
        provider, model_name = provider_model.split("$$$")
        sys_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "system", False)
        sys_sql3_table.set(
            "image-interpretation", {"provider": provider, "model": model_name}
        )

        await context.bot.send_message(
            chat_id=message.chat.id,
            text=f"Vision Interpretation model set to {provider} - {model_name}",
            parse_mode=ParseMode.HTML,
        )

    logger.info("Released lock for user: %s", identifier)


async def pending_user_handler(update: Update, context: CallbackContext) -> None:
    logger = logging.getLogger(__name__)
    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_sa_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        user_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "user_profile", False)

        keys = user_sql3_table.keys()
        name_list = []
        for key in keys:
            user_data = user_sql3_table.get(key)
            if user_data["status"] == "inactive":
                name_list.append(f"- {user_data['username']} (ID: {key})")

        if len(name_list) == 0:
            await message.reply_text(text="No pending user.", parse_mode=ParseMode.HTML)
            return

        output_string = "\n".join(name_list)
        await message.reply_text(text=output_string, parse_mode=ParseMode.HTML)


async def allow_pending_user_handler(update: Update, context: CallbackContext) -> None:
    logger = logging.getLogger(__name__)
    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_sa_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        user_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "user_profile", False)
        txts = message.text.split(" ")
        logger.info("txts: %s", txts)
        if len(txts) == 2:
            id_str = txts[1]
            try:
                id_int = int(id_str)
                identifier = format_identifier(id_int)
                uprofile = user_sql3_table.get(identifier)
                if uprofile is None:
                    output_string = "Invalid ID."

                uprofile["status"] = "active"
                user_sql3_table.set(identifier, uprofile)
                output_string = f"Access granted to {uprofile['username']}"
            except Exception as e:
                logger.error("allow_pending_user_handler: %s", e)
                output_string = "Invalid ID."
        else:
            output_string = "Invalid input."
        await message.reply_text(text=output_string, parse_mode=ParseMode.HTML)
