"""
All the handlers for the bot.
"""

import os
import logging

# import json
from asyncio import Lock
from typing import Optional
from hashlib import md5
from datetime import datetime

# External Imports
import telegram
from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode
import chromadb

from llm_agent_toolkit import ResponseMode, ShortTermMemory  # type: ignore
from llm_agent_toolkit._util import TokenUsage

from llms import LLMFactory
from pygrl import BasicStorage, GeneralRateLimiter as grl

# Internal Imports
from custom_workflow import (
    call_ai_ops,
    call_cc,
    image_interpreter_pipeline as ii_pipeline,
    update_preference,
    process_audio_input,
    reply,
    compress_conv_hx,
)
from custom_library import store_to_drive, format_identifier, map_file_extension
from transcriber import TranscriberFactory
from mystorage import ChromaDBFactory, WebCache, SQLite3_Storage
import myconfig

logger = logging.getLogger(__name__)

# Global Variables
chat_memory: dict[str, ShortTermMemory] = {}  # Cleanup mechanism is needed!
user_locks: dict[str, Lock] = {}  # Cleanup mechanism is needed!


rl_storage = BasicStorage()
rate_limiter = grl(rl_storage, 1, 1, 100)  # maximum 1 request every 30 seconds
user_stats: dict[str, tuple[bool, str]] = {}  # Cleanup mechanism is needed!

main_vdb: chromadb.ClientAPI = ChromaDBFactory.get_instance(
    persist=True, persist_directory="/temp/vect"
)
web_db = WebCache(ttl=600, maxsize=128)


#### Workflows ####


def register_user(
    identifier: str, username: str, force: bool = False, premium: bool = False
) -> None:
    user_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "user_profile", False)
    uprofile = user_sql3_table.get(identifier)
    if uprofile is None or force:
        sys_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "system", False)
        cc_row = sys_sql3_table.get("chat-completion")
        # assert cc_row is not None
        ii_row = sys_sql3_table.get("image-interpretation")
        # assert ii_row is not None
        user_sql3_table.set(
            identifier,
            {
                "username": username,
                "platform_t2t": cc_row["provider"],
                "model_t2t": cc_row["model_name"],
                "platform_i2t": ii_row["provider"],
                "model_i2t": ii_row["model_name"],
                "character": myconfig.DEFAULT_CHARACTER,
                "memory": None,
                "auto_routing": True,
                "usage": {"openai": 0, "ollama": 0, "gemini": 0, "deepseek": 0},
                "status": "active" if premium else "inactive",
            },
        )


def register_memory(identifier: str, force: bool = False) -> None:
    global chat_memory

    umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
    if umemory is None or force:
        chat_memory[identifier] = ShortTermMemory(max_entry=myconfig.MEMORY_LEN + 5)


def get_user_lock(identifier: str) -> Lock:
    """
    Get a lock for a specific user based on their identifier.

    Args:
        identifier (str): The identifier of the user.

    Returns:
        asyncio.Lock: The lock associated with the user.
    """
    global user_locks
    if identifier not in user_locks:
        user_locks[identifier] = Lock()
    return user_locks[identifier]


async def redirect(
    update: Update, context: CallbackContext, message: telegram.Message
) -> None:
    prompt: str = message.text
    if prompt.startswith("/help"):
        await help_handler(update, context)
    elif prompt.startswith("/start"):
        await start_handler(update, context)
    elif prompt.startswith("/usage"):
        await show_usage(update, context)
    elif prompt.startswith("/chat_model"):
        await show_chat_model_menu(update, context)
    elif prompt.startswith("/mycharacter"):
        await show_character_handler(update, context)
    elif prompt.startswith("/new"):
        await reset_chatmemory_handler(update, context)
    elif prompt.startswith("/clear"):
        await reset_user_handler(update, context)
    else:
        await message.reply_text(text="Invalid command.", parse_mode=ParseMode.HTML)
    return None


def generate_unique_filename(seed: str, extension: str, deterministic: bool = False):
    if not isinstance(seed, str):
        content = str(seed)
    else:
        content = seed

    if not deterministic:
        content = content + "_" + str(datetime.now().timestamp())

    hash_value = md5(content.encode()).hexdigest()
    return f"{hash_value}.{extension}"


def remove_related_to_file(mem: ShortTermMemory, file_prefix: str) -> ShortTermMemory:
    new_mem = ShortTermMemory(max_entry=myconfig.MEMORY_LEN + 5)

    for conv in mem.to_list():
        if conv["role"] == "user" and conv["content"].startswith(file_prefix):
            logger.warning("SKIP %s", conv["content"])
            continue
        new_mem.push(conv)

    return new_mem


#### Commands ####


async def middleware_function(update: Update, context: CallbackContext) -> None:
    """
    Intercepts messages
    """
    global chat_memory, rate_limiter, user_stats, main_vdb, web_db
    # logger = logging.getLogger(__name__)
    logger.info("Middleware => Update: %s", update)
    # Filter out Non-user updates
    channel_post = getattr(update, "channel_post", None)
    if channel_post:
        await context.bot.send_message(
            chat_id=channel_post.chat.id, text="This bot does not support `Channel`."
        )
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = format_identifier(message.chat.id)
    ulock = get_user_lock(identifier)
    if ulock.locked():
        await message.reply_text("Previous request is still running.")

    async with ulock:
        allowed_to_pass = rate_limiter.check_limit(identifier)
        user_stats[identifier] = (
            allowed_to_pass,
            "OK" if allowed_to_pass else "Exceed Rate Limit",
        )
        if not allowed_to_pass:
            return

        sys_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "system", False)
        emb_row = sys_sql3_table.get("embedding")
        assert emb_row is not None

        llm_factory = LLMFactory(
            vdb=main_vdb,
            webcache=web_db,
            encoder_config={
                "provider": emb_row["provider"],
                "model_name": emb_row["model_name"],
                "dimension": emb_row["dimension"],
            },
        )

        user_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "user_profile", False)
        # Authentication
        user_profile = user_sql3_table.get(identifier)
        if user_profile is None:
            if message.chat.id not in myconfig.PREMIUM_MEMBERS:
                register_memory(identifier, force=False)
                register_user(
                    identifier, message.from_user.name, force=True, premium=False
                )
                logger.warning("Unauthorized Access: %d", message.chat.id)
                context.user_data["access"] = "Unauthorized Access"
                await context.bot.send_message(
                    chat_id=message.chat.id, text="Unauthorized Access."
                )
                return
        elif (
            user_profile["status"] == "active"
            and context.user_data.get("access", None) == "Unauthorized Access"
        ):
            context.user_data["access"] = "Granted"
        elif user_profile["status"] == "denied":
            context.user_data["access"] = "Unauthorized Access"
            await context.bot.send_message(
                chat_id=message.chat.id, text="You have been denied access."
            )
            return

        register_memory(identifier, force=False)
        register_user(identifier, message.from_user.name, force=False, premium=True)

        cc_row = sys_sql3_table.get("chat-completion")
        t_row = sys_sql3_table.get("audio-transcription")
        assert cc_row is not None and t_row is not None

        umemory = chat_memory[identifier]
        uprofile = user_sql3_table.get(identifier)
        token_usage = TokenUsage(input_tokens=0, output_tokens=0)
        if message.text:
            if not message.text.startswith("/"):
                # Compress long conversations
                compressor = llm_factory.create_chat_llm(
                    cc_row["provider"], cc_row["model_name"], "compressor", False
                )
                umemory, compress_token_usage = await compress_conv_hx(
                    compressor, umemory
                )
                umemory.push({"role": "user", "content": message.text})
                if compress_token_usage.total_tokens > 0:
                    token_usage += compress_token_usage
                # Extract User Preference
                agent_zero = llm_factory.create_chat_llm(
                    cc_row["provider"], cc_row["model_name"], "extractor", True
                )
                current_preference = uprofile.get("memory", None)
                updated_preference, _token_usage = await update_preference(
                    agent_zero, umemory, current_preference
                )
                if _token_usage.total_tokens > 0:
                    uprofile["memory"] = updated_preference
                    token_usage += _token_usage
        elif message.voice or message.audio:
            transcriber_factory = TranscriberFactory(
                provider=t_row["provider"],
                model_name=t_row["model_name"],
                output_directory="/temp",
                audio_parameter=None,
            )
            transcript, txt_path = await process_audio_input(
                message=message,
                context=context,
                transcriber=transcriber_factory.get_transcriber(),
                user_folder=f"/temp/{identifier}",
            )

            if transcript is None:
                umemory.push(
                    {
                        "role": "user",
                        "content": "Audio Upload failed - File is too big.",
                    }
                )
            else:
                prefix = "Audio Upload" if message.audio else "Voice Input"
                content = f"{prefix}:\n{transcript}"
                umemory.push({"role": "user", "content": content})
                logger.info("Add %s to memory.", content)
                # Extract User Preference
                agent_zero = llm_factory.create_chat_llm(
                    cc_row["provider"], cc_row["model_name"], "extractor", True
                )
                current_preference = uprofile.get("memory", None)
                updated_preference, _token_usage = await update_preference(
                    agent_zero, umemory, current_preference
                )
                if _token_usage.total_tokens > 0:
                    uprofile["memory"] = updated_preference
                    token_usage += _token_usage

                await reply(
                    message, f"**Whisper 🎤 [{t_row['model_name']}]**:\n{transcript}"
                )

            if txt_path:
                txt_filename = os.path.basename(txt_path)
                await message.reply_document(
                    document=txt_path,
                    caption=message.caption,
                    allow_sending_without_reply=True,
                    filename=txt_filename,
                )
        # Clean Up
        ## Update Token Usage
        if token_usage.total_tokens > 0:
            uprofile["usage"][cc_row["provider"]] += token_usage.total_tokens
        ## Update User Profile
        user_sql3_table.set(identifier, uprofile)
        ## Update Chat Memory
        chat_memory[identifier] = umemory


async def show_character_handler(update: Update, context: CallbackContext) -> None:
    """
    TODO: ONLY show MEMORY!!!
    """
    global user_stats

    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )
    ulock = get_user_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            return

        user_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "user_profile", False)
        uprofile = user_sql3_table.get(identifier)
        assert uprofile is not None

        memory = uprofile["memory"]
        if memory:
            output_string = f"*Memory*:\n```json\n{memory}\n```"
        else:
            output_string = "It takes a while to build user profile. "

        await reply(message, output_string)

    logger.info("Released lock for user: %s", identifier)


async def show_chat_model_menu(update: Update, context: CallbackContext) -> None:
    global user_stats

    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_user_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

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
                            callback_data=f"set_chat_model|{provider}$$${model_name}",
                        )
                    ]
                )

        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        await message.reply_text(
            output_string, reply_markup=reply_markup, parse_mode=ParseMode.HTML
        )


async def show_vision_model_menu(update: Update, context: CallbackContext) -> None:
    global user_stats

    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_user_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

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
                            callback_data=f"set_vision_model|{provider}$$${model_name}",
                        )
                    ]
                )

        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        await message.reply_text(
            output_string, reply_markup=reply_markup, parse_mode=ParseMode.HTML
        )


async def set_chat_model_handler(update: Update, context: CallbackContext) -> None:
    global user_stats

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

    ulock = get_user_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        query = update.callback_query
        await query.answer()

        provider_model = query.data.split("|")[1]
        provider, model_name = provider_model.split("$$$")
        user_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "user_profile", False)
        uprofile = user_sql3_table.get(identifier)
        assert uprofile is not None
        uprofile["platform_t2t"] = provider
        uprofile["model_t2t"] = model_name

        user_sql3_table.set(identifier, uprofile)

        await context.bot.send_message(
            chat_id=message.chat.id,
            text=f"Chat Completion model set to {provider} - {model_name}",
            parse_mode=ParseMode.HTML,
        )

    logger.info("Released lock for user: %s", identifier)


async def set_vision_model_handler(update: Update, context: CallbackContext) -> None:
    global user_stats

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

    ulock = get_user_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        query = update.callback_query
        await query.answer()

        provider_model = query.data.split("|")[1]
        provider, model_name = provider_model.split("$$$")
        user_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "user_profile", False)
        uprofile = user_sql3_table.get(identifier)
        assert uprofile is not None
        uprofile["platform_i2t"] = provider
        uprofile["model_i2t"] = model_name

        user_sql3_table.set(identifier, uprofile)

        await context.bot.send_message(
            chat_id=message.chat.id,
            text=f"Vision Interpretation model set to {provider} - {model_name}",
            parse_mode=ParseMode.HTML,
        )

    logger.info("Released lock for user: %s", identifier)


async def help_handler(update: Update, context: CallbackContext) -> None:
    repo_path: str = myconfig.REPO_PATH
    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    # identifier = format_identifier(message.chat.id)
    await message.reply_text(
        f"Thank you for trying out my project. You can find me at GitHub: {repo_path}"
    )


async def start_handler(update: Update, context: CallbackContext) -> None:
    repo_path: str = myconfig.REPO_PATH

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    await message.reply_text(
        f"Thank you for trying out my project. You can find me at GitHub: {repo_path}"
    )


async def error_handler(update: object, context: CallbackContext):
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    logger.info("\nError Handler => Update: %s", update)


async def reset_chatmemory_handler(update: Update, context: CallbackContext):
    global chat_memory, user_locks, user_stats
    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = format_identifier(message.chat.id)

    ulock = get_user_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        umemory = chat_memory.get(identifier, None)
        if umemory is None:
            raise ValueError("Memory is None.")

        umemory.clear()
        await message.reply_text("Memory has been reset.")

    logger.info("Released lock for user: %s", identifier)


async def reset_user_handler(update: Update, context: CallbackContext):
    global user_locks, user_stats
    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = format_identifier(message.chat.id)

    ulock = get_user_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        register_memory(identifier, force=True)
        register_user(identifier, message.from_user.name, force=True)
        await message.reply_text("User has been reset.")
    logger.info("Released lock for user: %s", identifier)


async def show_usage(update: Update, context: CallbackContext) -> None:
    """
    Issue: Misalignment when use default font in Telegram Client.
    """
    global user_stats
    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = format_identifier(message.chat.id)

    ulock = get_user_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        user_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "user_profile", False)
        uprofile = user_sql3_table.get(identifier)
        logger.info("Profile: %s", uprofile)
        assert uprofile is not None

        output_strings = ["**Usage Report**"]

        usage_tracking = uprofile["usage"]
        for k, v in usage_tracking.items():
            output_strings.append(f">> *{k:<15s}*: {v}")

        await reply(message, "\n".join(output_strings))

    logger.info("Released lock for user: %s", identifier)


async def delete_file_handler(update: Update, context: CallbackContext) -> None:
    from llm_agent_toolkit.encoder import OllamaEncoder, OpenAIEncoder
    from llm_agent_toolkit.chunkers import FixedGroupChunker
    from llm_agent_toolkit.memory import ChromaMemory

    global user_stats, chat_memory
    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = format_identifier(message.chat.id)
    ulock = get_user_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        reply_to_message: Optional[telegram.Message] = getattr(
            message, "reply_to_message", None
        )
        if reply_to_message is None:
            await reply(message, "Please REPLY to a specific file")
            return None

        target_document: Optional[telegram.Document] = getattr(
            reply_to_message, "document", None
        )
        if target_document is None:
            await reply(message, "Please REPLY to a specific file")
            return None

        user_vdb: chromadb.ClientAPI = ChromaDBFactory.get_instance(
            persist=True, persist_directory=f"/temp/vect/{identifier}"
        )
        sys_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "system", False)
        e_row = sys_sql3_table.get("embedding")
        if e_row["provider"] == "local":
            encoder = OllamaEncoder(
                myconfig.OLLAMA_HOST, model_name=e_row["model_name"]
            )
        elif e_row["provider"] == "openai":
            encoder = OpenAIEncoder(
                model_name=e_row["model_name"], dimension=e_row["dimension"]
            )
        else:
            # encoder = TransformerEncoder(
            #     model_name=e_row["model_name"], directory="/temp"
            # )
            raise RuntimeError("Selected unsupported embedding provider.")

        chunker_config = {"K": 1}
        chunker = FixedGroupChunker(config=chunker_config)
        cm = ChromaMemory(vdb=user_vdb, encoder=encoder, chunker=chunker)

        try:
            cm.delete(identifier=target_document.file_name)
            umemory = chat_memory.get(identifier, None)
            chat_memory[identifier] = remove_related_to_file(
                umemory, target_document.file_name
            )
            await reply(message, "File has been deleted.")
        except Exception as e:
            logger.error(
                "Fail to delele file %s: %s",
                target_document.file_name,
                str(e),
                exc_info=True,
                stack_info=True,
            )
            await reply(message, "Failed to delete file.")

        return None


#### Inputs ####


async def message_handler(update: Update, context: CallbackContext) -> None:
    global chat_memory, user_locks, user_stats

    if context.user_data.get("access", None) == "Unauthorized Access":
        return None
    # We can consider to redirect these requests
    # Temporary turn off
    # channel_post = getattr(update, "channel_post", None)
    # if channel_post:
    #     await redirect(update, context, channel_post)
    #     return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = format_identifier(message.chat.id)
    ulock = get_user_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)

        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        user_vdb: chromadb.ClientAPI = ChromaDBFactory.get_instance(
            persist=True, persist_directory=f"/temp/vect/{identifier}"
        )
        sys_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "system", False)
        emb_row = sys_sql3_table.get("embedding")
        assert emb_row is not None

        llm_factory = LLMFactory(
            vdb=main_vdb,
            webcache=web_db,
            user_vdb=user_vdb,
            encoder_config={
                "provider": emb_row["provider"],
                "model_name": emb_row["model_name"],
                "dimension": emb_row["dimension"],
            },
        )

        prompt: str = message.text
        if prompt is None or prompt == "":
            raise ValueError("Prompt is None or empty.")

        umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
        if umemory is None:
            raise ValueError("Memory is None.")

        assert prompt == umemory.to_list()[-1]["content"]

        final_response_content = await call_ai_ops(
            message, identifier, chat_memory, llm_factory
        )
        # Output generated response
        await reply(message, final_response_content)
    logger.info("Released lock for user: %s", identifier)


async def photo_handler(update: Update, context: CallbackContext):
    global chat_memory, user_locks, user_stats, main_vdb, web_db

    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = format_identifier(message.chat.id)

    ulock = get_user_lock(identifier)
    async with ulock:
        # logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        user_vdb: chromadb.ClientAPI = ChromaDBFactory.get_instance(
            persist=True, persist_directory=f"/temp/vect/{identifier}"
        )
        sys_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "system", False)
        emb_row = sys_sql3_table.get("embedding")
        assert emb_row is not None

        llm_factory = LLMFactory(
            vdb=main_vdb,
            webcache=web_db,
            user_vdb=user_vdb,
            encoder_config={
                "provider": emb_row["provider"],
                "model_name": emb_row["model_name"],
                "dimension": emb_row["dimension"],
            },
        )

        umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
        if umemory is None:
            raise ValueError("Memory is None.")

        user_folder = f"/temp/{identifier}"
        if not os.path.exists(user_folder):
            os.mkdir(user_folder)

        photo = message.photo[-1]

        file_id: str = photo.file_id
        export_path = os.path.join(
            user_folder, generate_unique_filename(file_id, "jpg", True)
        )
        # logger.info("Loading %s from Telegram's Server", file_id)
        await store_to_drive(file_id, export_path, context)
        # logger.info("Stored %s", export_path)
        if message.caption:
            prompt = f"<caption>{message.caption}</caption>"
        else:
            prompt = "Describe the image."

        umemory.push({"role": "user", "content": prompt})

        user_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "user_profile", False)
        uprofile = user_sql3_table.get(identifier)
        assert uprofile is not None

        ii_resp_tuple = await ii_pipeline(uprofile, prompt, export_path, llm_factory)
        output_string, uprofile = ii_resp_tuple
        # platform = uprofile["platform_i2t"]
        # uprofile["usage"][platform] += usage.total_tokens
        umemory.push({"role": "user", "content": output_string})

        await reply(message, output_string)

        prompt = output_string
        if message.caption:
            prompt += f"\nCaption={message.caption}"

        generated_content = await call_ai_ops(
            message, identifier, chat_memory, llm_factory
        )
        await reply(message, generated_content)
    logger.info("Released lock for user: %s", identifier)


async def voice_handler(update: Update, context: CallbackContext) -> None:
    global chat_memory, user_locks, user_stats, main_vdb, web_db

    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = format_identifier(message.chat.id)

    ulock = get_user_lock(identifier)
    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        user_vdb: chromadb.ClientAPI = ChromaDBFactory.get_instance(
            persist=True, persist_directory=f"/temp/vect/{identifier}"
        )
        sys_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "system", False)
        emb_row = sys_sql3_table.get("embedding")
        assert emb_row is not None

        llm_factory = LLMFactory(
            vdb=main_vdb,
            webcache=web_db,
            user_vdb=user_vdb,
            encoder_config={
                "provider": emb_row["provider"],
                "model_name": emb_row["model_name"],
                "dimension": emb_row["dimension"],
            },
        )

        umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
        if umemory is None:
            raise ValueError("Memory is None.")

        generated_content = await call_ai_ops(
            message, identifier, chat_memory, llm_factory
        )
        await reply(message, generated_content)
    logger.info("Released lock for user: %s", identifier)


async def audio_handler(update: Update, context: CallbackContext) -> None:
    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    # identifier: str = format_identifier(message.chat.id)

    # ulock = get_user_lock(identifier)
    # await reply(message, "COPY")


async def document_handler(update: Update, context: CallbackContext) -> None:
    from llm_agent_toolkit.loader import (
        TextLoader,
        PDFLoader,
        MsWordLoader,
        ImageToTextLoader,
    )
    from llm_agent_toolkit.encoder import OllamaEncoder, OpenAIEncoder
    from llm_agent_toolkit.memory import ChromaMemory
    from llm_agent_toolkit.chunkers import SemanticChunker

    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = format_identifier(message.chat.id)

    ulock = get_user_lock(identifier)

    async with ulock:
        # logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        user_vdb: chromadb.ClientAPI = ChromaDBFactory.get_instance(
            persist=True, persist_directory=f"/temp/vect/{identifier}"
        )
        sys_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "system", False)
        emb_row = sys_sql3_table.get("embedding")
        assert emb_row is not None

        llm_factory = LLMFactory(
            vdb=main_vdb,
            webcache=web_db,
            user_vdb=user_vdb,
            encoder_config={
                "provider": emb_row["provider"],
                "model_name": emb_row["model_name"],
                "dimension": emb_row["dimension"],
            },
        )

        umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
        if umemory is None:
            raise ValueError("Memory is None.")

        namespace = f"/temp/{identifier}"
        fid = message.document.file_id

        f_ext = map_file_extension(message.document.mime_type)
        f_name = message.document.file_name
        # assert f_ext == f_name.split(".")[-1], f"{f_ext} != {f_name.split('.')[-1]}"
        export_path = f"{namespace}/{f_name}"
        is_new_upload = not os.path.exists(export_path)
        await store_to_drive(fid, export_path, context, overwrite=True)

        db = SQLite3_Storage(myconfig.DB_PATH, "user_profile", False)
        uprofile = db.get(identifier)

        character = "speed_reader"
        if f_ext in ["txt", "md"]:
            loader = TextLoader()
        else:
            assert uprofile is not None
            ii = llm_factory.create_image_interpreter(
                platform=uprofile["platform_i2t"],
                model_name=uprofile["model_i2t"],
                system_prompt=myconfig.CHARACTER["seer"]["system_prompt"],
                temperature=myconfig.CHARACTER["seer"]["temperature"],
            )
            if f_ext in ["jpg", "png", "jpeg"]:
                character = "general"
                loader = ImageToTextLoader(
                    image_interpreter=ii,
                    prompt=(
                        message.caption if message.caption else "Describe the image."
                    ),
                )
            elif f_ext == "pdf":
                loader = PDFLoader(
                    text_only=False, tmp_directory=namespace, image_interpreter=ii
                )
            elif f_ext == "docx":
                loader = MsWordLoader(
                    text_only=False, tmp_directory=namespace, image_interpreter=ii
                )
            else:
                raise ValueError("Invalid file ext")

        sys_sql3_table = SQLite3_Storage(myconfig.DB_PATH, "system", False)
        e_row = sys_sql3_table.get("embedding")
        content: str = await loader.load_async(export_path)
        if e_row["provider"] == "local":
            encoder = OllamaEncoder(
                myconfig.OLLAMA_HOST, model_name=e_row["model_name"]
            )
        elif e_row["provider"] == "openai":
            encoder = OpenAIEncoder(
                model_name=e_row["model_name"], dimension=e_row["dimension"]
            )
        else:
            # encoder = TransformerEncoder(
            #     model_name=e_row["model_name"], directory="/temp"
            # )
            raise RuntimeError("Selected unsupported embedding provider.")

        K = max(len(content) // min(encoder.ctx_length, 600), 1)
        chunker_config = {
            "K": K,
            "MAX_ITERATION": K * 5,
            "update_rate": 0.1,
            "min_coverage": 0.95,
        }
        chunker = SemanticChunker(encoder=encoder, config=chunker_config)
        cm = ChromaMemory(vdb=user_vdb, encoder=encoder, chunker=chunker)
        added = False
        try:
            if is_new_upload:
                cm.add(
                    document_string=content,
                    identifier=f_name,
                    metadata={"mime_type": message.document.mime_type},
                )
            else:
                cm.update(
                    identifier=f_name,
                    document_string=content,
                    metadata={"mime_type": message.document.mime_type},
                )
            added = True
        except ValueError as ve:
            logger.error("Add data to ChromaMemory: FAILED.\n%s", str(ve))
        if not added:
            output_string = "Add data to ChromaMemory: *FAILED*"
            await reply(message, output_string)
            return None

        llm = llm_factory.create_chat_llm(
            uprofile["platform_t2t"], uprofile["model_t2t"], character, False
        )
        prompt = """
        ---
        CONTENT START
        ${{CONTENT}}
        CONTENT END
        ---
        """
        content_window = int(min(llm.context_length * 0.75, 20_000))
        templated_prompt = prompt.replace("${{CONTENT}}", content[:content_window])
        responses, usage = await call_cc(
            llm,
            prompt=templated_prompt,
            context=None,
            mode=ResponseMode.DEFAULT,
            response_format=None,
        )
        file_summary = responses[-1]["content"]

        umemory.push({"role": "user", "content": f"{f_name}:\n{file_summary}"})
        chat_memory[identifier] = umemory

        logger.info("Generated %d responses.", len(responses))
        uprofile["usage"][uprofile["platform_t2t"]] += usage.total_tokens
        db.set(identifier, uprofile)
        character_name = myconfig.CHARACTER[character]["name"]
        model_name = uprofile["model_t2t"]
        output_string = f"**{character_name}** [*{model_name}*]:\n{file_summary}"
        await reply(message, output_string)


async def attachment_handler(update: Update, context: CallbackContext) -> None:
    if context.user_data.get("access", None) == "Unauthorized Access":
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    # identifier: str = format_identifier(message.chat.id)
    # ulock = get_user_lock(identifier)

    if message.photo:
        logger.info("Redirect -> photo_handler")
        await photo_handler(update, context)
        return None

    if message.document:
        f_ext = map_file_extension(message.document.mime_type)
        # Excel, CSV, JSON, and other structured data files should be handled differently

        # Focus on text content first
        if f_ext not in ["pdf", "docx", "txt", "md", "html", "jpg", "png", "jpeg"]:
            logger.warning("Unsupported File Extension: %s", f_ext)
            return None

        logger.info("Redirect -> document_handler")
        await document_handler(update, context)
        return None
