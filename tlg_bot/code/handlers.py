"""
All the handlers for the bot.
"""

import os
import logging

# import json
import re
from asyncio import Lock
from typing import Any, Optional
from hashlib import md5
from datetime import datetime

# External Imports
import telegram
from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode
from pydantic import BaseModel
import chromadb

from llm_agent_toolkit import Core, ResponseMode, ShortTermMemory  # type: ignore
from llm_agent_toolkit._util import TokenUsage

from llms import LLMFactory
from pygrl import BasicStorage, GeneralRateLimiter as grl

# Internal Imports
from custom_workflow import (
    image_interpreter_pipeline as ii_pipeline,
    find_best_agent,
    update_memory,
    process_audio_input,
)
from custom_library import store_to_drive, format_identifier
from transcriber import TranscriberFactory
from mystorage import ChromaDBFactory, WebCache, SQLite3_Storage
import myconfig

logger = logging.getLogger(__name__)

# Global Variables
chat_memory: dict[str, ShortTermMemory] = {}
user_locks: dict[str, Lock] = {}
db = SQLite3_Storage("/db/ost.db", "user_profile", False)

rl_storage = BasicStorage()
rate_limiter = grl(rl_storage, 1, 1, 100)
user_stats: dict[str, tuple[bool, str]] = {}

main_vdb: chromadb.ClientAPI = ChromaDBFactory.get_instance(
    persist=True, persist_directory="/temp/vect"
)
web_db = WebCache(ttl=600, maxsize=128)
llm_factory = LLMFactory(vdb=main_vdb, webcache=web_db)
transcriber_factory = TranscriberFactory(
    provider="local",
    model_name="turbo",
    output_directory="/temp",
    audio_parameter=None,
)


def register_user(identifier: str, force: bool = False) -> None:
    global db
    uprofile = db.get(identifier)
    if uprofile is None or force:
        pt2t, mt2t = myconfig.DEFAULT_T2T_MODEL
        pi2t, mi2t = myconfig.DEFAULT_I2T_MODEL
        db.set(
            identifier,
            {
                "platform_t2t": pt2t,
                "model_t2t": mt2t,
                "platform_i2t": pi2t,
                "model_i2t": mi2t,
                "character": myconfig.DEFAULT_CHARACTER,
                "creativity": 0.7,
                "memory": None,
                "auto_routing": True,
                "usage": {"openai": 0, "ollama": 0, "gemini": 0, "deepseek": 0},
            },
        )


def register_memory(identifier: str, force: bool = False) -> None:
    global chat_memory

    umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
    if umemory is None or force:
        chat_memory[identifier] = ShortTermMemory(max_entry=myconfig.MEMORY_LEN + 5)


async def show_character_handler(update: Update, context: CallbackContext) -> None:
    """
    TODO: ONLY show MEMORY!!!
    """
    global db, user_stats
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

        uprofile = db.get(identifier)
        assert uprofile is not None

        memory = uprofile["memory"]
        if memory:
            output_string = f"*Memory*:\n```json\n{memory}\n```"
        else:
            output_string = "It takes a while to build user profile. "

        await reply(message, output_string)

    logger.info("Released lock for user: %s", identifier)


async def show_model_menu(update: Update, context: CallbackContext) -> None:
    global user_stats

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

        output_string = "Click to select a model:\n"

        keyboard = []
        providers = list(myconfig.PROVIDER.keys())
        for provider in providers:
            models = list(myconfig.PROVIDER[provider]["t2t"])
            for model_name in models:
                name = f"{provider} - {model_name}"
                keyboard.append(
                    [
                        telegram.InlineKeyboardButton(
                            name, callback_data=f"set_model|{provider}$$${model_name}"
                        )
                    ]
                )

        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        await message.reply_text(
            output_string, reply_markup=reply_markup, parse_mode=ParseMode.HTML
        )


async def set_model_handler(update: Update, context: CallbackContext) -> None:
    global db, user_stats

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
        uprofile = db.get(identifier)
        assert uprofile is not None
        uprofile["platform_t2t"] = provider
        uprofile["model_t2t"] = model_name

        db.set(identifier, uprofile)

        await context.bot.send_message(
            chat_id=message.chat.id,
            text=f"Chat Completion set to {provider} - {model_name}",
            parse_mode=ParseMode.HTML,
        )

    logger.info("Released lock for user: %s", identifier)


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


async def reply(
    message: telegram.Message, output_string: str
) -> telegram.Message | None:
    from custom_library import escape_html, escape_markdown_extended

    if myconfig.DEBUG == "1":
        logger.info(">> %s", output_string)

    MSG_MAX_LEN = 3800
    sections = re.split(r"\n{2,}", output_string)
    current_chunk = ""
    msg = None
    for section in sections:
        if len(current_chunk) + len(section) + 2 <= MSG_MAX_LEN:
            current_chunk += section + "\n\n"
        else:
            try:
                html_compatible_chunk = escape_html(current_chunk)
                msg = await message.reply_text(
                    html_compatible_chunk, parse_mode=ParseMode.HTML
                )
            except telegram.error.BadRequest as bre:
                if "Can't parse entities:" not in str(bre):
                    logger.error(str(bre))
                    raise

                logger.warning("Fallback to MARKDOWN_V2 mode.")
                formatted_chunk = escape_markdown_extended(current_chunk)
                # formatted_chunk, is_close = handle_triple_ticks(formatted_chunk, is_close)
                msg = await message.reply_text(
                    formatted_chunk, parse_mode=ParseMode.MARKDOWN_V2
                )
            current_chunk = section + "\n\n"

    # Handle the last chunk
    if current_chunk:
        try:
            html_compatible_chunk = escape_html(current_chunk)
            msg = await message.reply_text(
                html_compatible_chunk, parse_mode=ParseMode.HTML
            )
        except telegram.error.BadRequest as bre:
            if "Can't parse entities:" not in str(bre):
                logger.error(str(bre))
                raise

            logger.warning("Fallback to MARKDOWN_V2 mode.")
            formatted_chunk = escape_markdown_extended(current_chunk)
            # formatted_chunk, _ = handle_triple_ticks(formatted_chunk, is_close)
            msg = await message.reply_text(
                formatted_chunk, parse_mode=ParseMode.MARKDOWN_V2
            )
    return msg


class Counter:
    def __init__(self):
        self.usage = 0

    def increment(self):
        self.usage += 1

    def get(self):
        return self.usage


x = """
Update(
    channel_post=Message(
        channel_chat_created=False, 
        chat=Chat(id=-1002303276058, title='JW', type=<ChatType.CHANNEL>, username='whaler_bot_dev'), 
        date=datetime.datetime(2025, 3, 6, 8, 39, tzinfo=datetime.timezone.utc), 
        delete_chat_photo=False, 
        group_chat_created=False, 
        message_id=2, 
        sender_chat=Chat(id=-1002303276058, title='JW', type=<ChatType.CHANNEL>, username='whaler_bot_dev'), 
        supergroup_chat_created=False, text='pop'
    ), 
    update_id=465202143
)
"""


async def middleware_function(update: Update, context: CallbackContext) -> None:
    """
    Intercepts messages
    """
    global chat_memory, rate_limiter, db, user_stats, llm_factory, transcriber_factory
    # logger = logging.getLogger(__name__)
    logger.info("Middleware => Update: %s", update)
    # Filter out Non-user updates
    channel_post = getattr(update, "channel_post", None)
    if channel_post:
        chat_id = channel_post.chat.id
        await context.bot.send_message(
            chat_id=chat_id, text="This bot does not support `Channel`."
        )
        return None

    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier: str = format_identifier(message.chat.id)

    user_data = getattr(context, "user_data", None)
    if user_data is None:
        logger.warning(">> %s", update)
        context.chat_data.setdefault(identifier, "BLOCKED")
        raise RuntimeError("Non-user updates")
    # user_stats: dict[str, tuple[bool, str]] = context.bot_data.get("user_stats")
    if message.chat.id not in myconfig.PREMIUM_MEMBERS:
        logger.warning("Unauthorized Access: %d", message.chat.id)
        user_stats[identifier] = (False, "Unauthorized Access")
        return

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

        register_user(identifier)
        register_memory(identifier)
        token_usage = None
        agent_zero = llm_factory.create_chat_llm(
            *myconfig.DEFAULT_T2T_MODEL, "extractor", 0.3, False
        )
        if message.text:
            if not message.text.startswith("/"):
                token_usage = await update_memory(
                    agent_zero, chat_memory, db, identifier, message.text
                )
        elif message.voice or message.audio:
            transcript, txt_path = await process_audio_input(
                message=message,
                context=context,
                transcriber=transcriber_factory.get_transcriber(),
                user_folder=f"/temp/{identifier}",
            )

            if transcript:
                prefix = "Audio Upload" if message.audio else "Voice Input"
                content = f"{prefix}:\n{transcript}"
                logger.info("Add %s to memory.", content)
                token_usage = await update_memory(
                    agent_zero, chat_memory, db, identifier, content
                )
                await reply(message, f"**Whisper 🎤**:\n{transcript}")

            if txt_path:
                await message.reply_document(
                    document=txt_path,
                    caption=message.caption,
                    allow_sending_without_reply=True,
                    filename=os.path.basename(txt_path),
                )

        if token_usage:
            uprofile = db.get(identifier)
            assert uprofile is not None
            default_provider, _ = myconfig.DEFAULT_T2T_MODEL
            uprofile["usage"][default_provider] += token_usage.total_tokens
            db.set(identifier, uprofile)


async def help_handler(update: Update, context: CallbackContext) -> None:
    repo_path: str = myconfig.REPO_PATH
    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    identifier = format_identifier(message.chat.id)
    status = context.chat_data.get(identifier, None)
    if status:
        logger.warning("%s => %s", identifier, status)

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


async def call_llm(
    llm: Core,
    prompt: str,
    context: list[dict],
    mode: ResponseMode,
    response_format: Any,
) -> tuple[list[dict], TokenUsage]:
    MAX_RETRY = 5
    iteration = 0
    while iteration < MAX_RETRY:
        logger.info("Attempt: %d", iteration)
        try:
            responses, usage = await llm.run_async(
                query=prompt, context=context, mode=mode, format=response_format
            )
            return responses, usage
        except ValueError as ve:
            if "max_output_tokens <= 0" in str(ve):
                context = context[1:]
                iteration += 1
            else:
                logger.error("call_llm: ValueError: %s", ve)
                iteration += 1
        except Exception as e:
            logger.error("call_llm: Exception: %s", e)
            raise
    raise ValueError(f"max_output_tokens <= 0. Retry up to {MAX_RETRY} times.")


async def call_chat_llm(
    profile: dict, memory: ShortTermMemory, prompt: str, tlg_msg: telegram.Message
) -> tuple[str, ShortTermMemory, dict]:
    global llm_factory

    logger.info("Executing call_chat_llm")
    platform = profile["platform_t2t"]
    model_name = profile["model_t2t"]
    character = profile["character"]
    metadata = profile["memory"]

    recent_conv = memory.last_n(myconfig.MEMORY_LEN)
    if len(recent_conv) > 2:
        recent_conv = recent_conv[:-1]
    else:
        recent_conv = None
    if metadata:
        if recent_conv:
            recent_conv.insert(
                0, {"role": "system", "content": f"<metadata>{metadata}</metadata>"}
            )
        else:
            recent_conv = [
                {"role": "system", "content": f"<metadata>{metadata}</metadata>"}
            ]

    auto_routing = profile["auto_routing"]
    if auto_routing:
        agent_router = llm_factory.create_chat_llm(
            *myconfig.DEFAULT_T2T_MODEL, "router", 0.3, True
        )
        tlg_msg = await tlg_msg.edit_text(
            "<b>Progress</b>: <i>ROUTING</i>", parse_mode=ParseMode.HTML
        )
        character, find_best_agent_usage = await find_best_agent(
            agent_router,
            prompt,
            context=recent_conv[-3:] if recent_conv else recent_conv,
        )
        profile["usage"][platform] += find_best_agent_usage.total_tokens

    llm = llm_factory.create_chat_llm(
        platform, model_name, character, profile["creativity"]
    )
    tlg_msg = await tlg_msg.edit_text(
        "<b>Progress</b>: <i>GENERATING</i>", parse_mode=ParseMode.HTML
    )
    responses, usage = await call_llm(
        llm, prompt, recent_conv, ResponseMode.DEFAULT, None
    )
    tlg_msg = await tlg_msg.edit_text(
        f"<b>Progress</b>: <i>DONE</i>\n<b>Usage</b>:\nInput: {usage.input_tokens}\nOutput: {usage.output_tokens}\nTotal: {usage.total_tokens}",
        parse_mode=ParseMode.HTML,
    )
    profile["usage"][platform] += usage.total_tokens

    logger.info("Generated %d responses.", len(responses))
    for response in responses:
        # logger.info("\nResponse: %s", response["content"])
        memory.push({"role": "assistant", "content": response["content"]})

    final_response = responses[-1]
    character_name = myconfig.CHARACTER[character]["name"]
    output_string = (
        f"**{character_name}** [*{model_name}*]:\n{final_response['content']}"
    )
    return output_string, memory, profile


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
    elif prompt.startswith("/model"):
        await show_model_menu(update, context)
    elif prompt.startswith("/mycharacter"):
        await show_character_handler(update, context)
    elif prompt.startswith("/new"):
        await reset_chatmemory_handler(update, context)
    elif prompt.startswith("/clear"):
        await reset_user_handler(update, context)
    else:
        await message.reply_text(text="Invalid command.", parse_mode=ParseMode.HTML)
    return None


async def message_handler(update: Update, context: CallbackContext) -> None:
    global chat_memory, user_locks, db, user_stats

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

        tlg_msg = await message.reply_text(
            "<b>Progress</b>: <i>START</i>", parse_mode=ParseMode.HTML
        )

        prompt: str = message.text
        if prompt is None or prompt == "":
            raise ValueError("Prompt is None or empty.")

        umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
        if umemory is None:
            raise ValueError("Memory is None.")

        uprofile = db.get(identifier)
        assert uprofile is not None

        output_string, updated_memory, updated_profile = await call_chat_llm(
            uprofile, umemory, prompt, tlg_msg
        )
        db.set(identifier, updated_profile)
        chat_memory[identifier] = updated_memory
        await reply(message, output_string)

    logger.info("Released lock for user: %s", identifier)


def generate_unique_filename(seed: str, extension: str, deterministic: bool = False):
    if not isinstance(seed, str):
        content = str(seed)
    else:
        content = seed

    if not deterministic:
        content = content + "_" + str(datetime.now().timestamp())

    hash_value = md5(content.encode()).hexdigest()
    return f"{hash_value}.{extension}"


async def photo_handler(update: Update, context: CallbackContext):
    global chat_memory, user_locks, db, user_stats

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

        tlg_msg = await message.reply_text("<b>START</b>", parse_mode=ParseMode.HTML)

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

        uprofile = db.get(identifier)
        assert uprofile is not None

        ii_resp_tuple = await ii_pipeline(uprofile, prompt, export_path, llm_factory)
        output_string, usage = ii_resp_tuple
        platform = uprofile["platform_i2t"]
        uprofile["usage"][platform] += usage.total_tokens
        umemory.push({"role": "user", "content": output_string})

        await reply(message, output_string)

        prompt = output_string
        if message.caption:
            prompt += f"\nCaption={message.caption}"

        response_tuple = await call_chat_llm(uprofile, umemory, prompt, tlg_msg)
        output_string, updated_memory, updated_profile = response_tuple
        # Leaving
        db.set(identifier, updated_profile)
        updated_memory.push({"role": "assistant", "content": output_string})
        chat_memory[identifier] = updated_memory
        await reply(message, output_string)
    # logger.info("Released lock for user: %s", identifier)


async def reset_chatmemory_handler(update: Update, context: CallbackContext):
    global chat_memory, user_locks, user_stats

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
        register_user(identifier, force=True)
        await message.reply_text("User has been reset.")
    logger.info("Released lock for user: %s", identifier)


class CompressContent(BaseModel):
    topics: list[str]
    compressed: str


async def compress_memory_handler(update: Update, context: CallbackContext):
    global chat_memory, user_locks, db, user_stats

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

        context = umemory.to_list()
        if len(context) <= 1:
            await message.reply_text("Memory is too short to compress.")
            return

        all_commands = True
        for c in context:
            if not c["content"].startswith("/"):
                all_commands = False
                break

        if all_commands:
            umemory.clear()
            await message.reply_text("Memory is too short to compress.")
            return

        context = context[:-1]
        prompt = "Compress the past conversations."

        uprofile = db.get(identifier)
        assert uprofile is not None

        platform = uprofile["platform_t2t"]
        model_name = uprofile["model_t2t"]
        llm = llm_factory.create_chat_llm(platform, model_name, "general", 0.0)

        responses, usage = await call_llm(
            llm, prompt, context, ResponseMode.DEFAULT, None
        )
        uprofile["usage"][platform] += usage.total_tokens
        db.set(identifier, uprofile)

        response = responses[0]
        content = response["content"]

        if content is None or content == "":
            raise ValueError("Content is None or empty.")

        umemory.clear()
        umemory.push({"role": "assistant", "content": content})

        await message.reply_text("Memory has been compressed.")

    logger.info("Released lock for user: %s", identifier)


async def voice_handler(update: Update, context: CallbackContext) -> None:
    global chat_memory, user_locks, db, user_stats

    logger.info("The VOICE!!!")
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

        umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
        if umemory is None:
            raise ValueError("Memory is None.")

        recent_conversation = umemory.last_n(n=myconfig.MEMORY_LEN)
        if len(recent_conversation) == 0:
            raise ValueError("Recent Conversation is None.")

        prompt = recent_conversation[-1]["content"]
        logger.info("Prompt: %s", prompt)
        if len(recent_conversation) > 1:
            recent_conversation = recent_conversation[:-1]  # The last one is the prompt
        else:
            recent_conversation = None

        uprofile = db.get(identifier)
        assert uprofile is not None

        tlg_msg = await message.reply_text("<b>START</b>", parse_mode=ParseMode.HTML)
        chat_resp_tuple = await call_chat_llm(uprofile, umemory, prompt, tlg_msg)
        output_string, updated_memory, updated_profile = chat_resp_tuple
        db.set(identifier, updated_profile)
        chat_memory[identifier] = updated_memory

        await reply(message, output_string)

    logger.info("Released lock for user: %s", identifier)


async def audio_handler(update: Update, context: CallbackContext) -> None:
    global chat_memory, user_locks, db, user_stats

    logger.info("The AUDIO!!!")
    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        return None

    # identifier: str = format_identifier(message.chat.id)

    # ulock = get_user_lock(identifier)
    # await reply(message, "COPY")


async def show_usage(update: Update, context: CallbackContext) -> None:
    """
    Issue: Misalignment when use default font in Telegram Client.
    """
    global user_stats, db

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

        uprofile = db.get(identifier)
        logger.info("Profile: %s", uprofile)
        assert uprofile is not None

        output_strings = ["**Usage Report**"]

        usage_tracking = uprofile["usage"]
        for k, v in usage_tracking.items():
            output_strings.append(f">> *{k:<15s}*: {v}")

        await reply(message, "\n".join(output_strings))

    logger.info("Released lock for user: %s", identifier)
