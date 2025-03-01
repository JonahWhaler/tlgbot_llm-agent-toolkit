"""
All the handlers for the bot.
"""

import os
import logging
import json
import re
from asyncio import Lock
from typing import Optional
from hashlib import md5
from datetime import datetime

import telegram
from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode
from pydantic import BaseModel

from llm_agent_toolkit import ResponseMode  # type: ignore
from llm_agent_toolkit._memory import ShortTermMemory  # type: ignore
from llm_agent_toolkit._core import ImageInterpreter
import chromadb
from llm_agent_toolkit.memory import ChromaMemory

from llms import LLMFactory, IIResponse
from pygrl import BasicStorage, GeneralRateLimiter as grl

from storage import SQLite3_Storage
import config
from config import (
    DEFAULT_T2T_MODEL,
    DEFAULT_I2T_MODEL,
    CHARACTER,
    DEFAULT_CHARACTER,
    PROVIDER,
    MEMORY_LEN,
    PREMIUM_MEMBERS,
    FREE,
)
from util import ChromaDBFactory

logger = logging.getLogger(__name__)

# Global Variables
chat_memory: dict[str, ShortTermMemory] = {}
user_locks: dict[str, Lock] = {}
db = SQLite3_Storage("/db/ost.db", "user_profile", False)

rl_storage = BasicStorage()
rate_limiter = grl(rl_storage, 1, 1, 100)
user_stats: dict[str, bool] = {}

llm_factory = LLMFactory()
agent_zero = llm_factory.create_chat_llm(
    "deepseek", "deepseek-chat", "extractor", 0.3, True
)

agent_router = llm_factory.create_chat_llm(
    "deepseek", "deepseek-chat", "router", 0.3, True
)
main_vdb: chromadb.ClientAPI = ChromaDBFactory.get_instance(
    persist=True, persist_directory="/temp/vect"
)


def register_user(identifier: str) -> None:
    global db
    uprofile = db.get(identifier)
    if uprofile:
        return

    pt2t, mt2t = DEFAULT_T2T_MODEL
    pi2t, mi2t = DEFAULT_I2T_MODEL
    uprofile = {
        "platform_t2t": pt2t,
        "model_t2t": mt2t,
        "platform_i2t": pi2t,
        "model_i2t": mi2t,
        "character": DEFAULT_CHARACTER,
        "creativity": 0.7,
        "memory": None,
        "auto_routing": True,
    }
    db.set(identifier, uprofile)

    return


def register_memory(identifier: str, force: bool = False) -> None:
    global chat_memory

    umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
    if umemory is None or force:
        chat_memory[identifier] = ShortTermMemory(max_entry=MEMORY_LEN + 5)


async def find_best_agent(prompt: str, context: list[dict] | None) -> str:
    logger.info("Routing...")
    global agent_router

    # Input
    agents = []
    for character in CHARACTER.keys():
        if CHARACTER[character]["io"] == "i2t":
            continue

        if CHARACTER[character]["access"] == "private":
            continue

        agent_object = {
            "name": character,
            "system_prompt": CHARACTER[character]["system_prompt"],
        }
        t = CHARACTER[character].get("tools", None)
        # heavily rely on the tool name, no further description here.
        if t:
            agent_object["tools"] = t
        agents.append(agent_object)

    input_prompt = {"request": prompt, "agents": agents}
    responses = await agent_router.run_async(
        query=json.dumps(input_prompt), context=context, mode=ResponseMode.JSON
    )

    # Output
    content = responses[0]["content"]
    try:
        output_object = json.loads(content)
        best_agent = output_object.get("agent", DEFAULT_CHARACTER)

        if best_agent not in CHARACTER.keys():
            logger.warning("Attempted to use unknown agent: %s", best_agent)
            logger.warning("Falling back to default agent.")
            best_agent = DEFAULT_CHARACTER
        reason = output_object.get("reason", "No reason provided.")
        logger.info("Pick %s. Reason: %s", CHARACTER[best_agent], reason)
        return best_agent
    except json.JSONDecodeError:
        return DEFAULT_CHARACTER


async def show_character_handler(update: Update, context: CallbackContext) -> None:
    global db
    message: Optional[telegram.Message] = getattr(update, "message", None)
    # edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)

    # is_edit: bool = False
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )
    ulock = get_user_lock(identifier)
    if ulock.locked():
        logger.info("Please wait for your previous request to finish.")
        return

    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)

        uprofile = db.get(identifier)
        assert uprofile is not None

        character = uprofile["character"]
        character_name = CHARACTER[character]["name"]
        system_prompt = CHARACTER[character]["system_prompt"]
        chatcompletion = uprofile["model_t2t"]
        memory = uprofile["memory"]

        creativity = uprofile["creativity"]

        if creativity == 0.0:
            creativity_level = "zero"
        elif creativity == 0.3:
            creativity_level = "low"
        elif creativity == 0.7:
            creativity_level = "medium"
        elif creativity == 1.0:
            creativity_level = "high"
        else:  # creativity == 1.2:
            creativity_level = "extreme"

        output_string = f"*Character*: {character_name} \n\n*AI Model*: {chatcompletion} \n*Creativity*: {creativity_level} \n\n"
        if len(system_prompt) > 500:
            output_string += (
                "*System Prompt*:\n```\n" + system_prompt[:500] + "...\n```"
            )
        else:
            output_string += "*System Prompt*:\n```\n" + system_prompt + "\n```"

        if memory:
            output_string += "\n\n*Memory*:\n```\n" + memory + "\n```"

        await reply(message, output_string)

    logger.info("Released lock for user: %s", identifier)


async def show_creativity_menu(update: Update, context: CallbackContext) -> None:
    message: Optional[telegram.Message] = getattr(update, "message", None)
    # edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)

    # is_edit: bool = False
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")

    output_string = "Click to select a level:\n"

    keyboard = []
    for level in ["zero", "low", "medium", "high", "extreme"]:
        keyboard.append(
            [
                telegram.InlineKeyboardButton(
                    level, callback_data=f"set_creativity|{level}"
                )
            ]
        )

    reply_markup = telegram.InlineKeyboardMarkup(keyboard)
    await message.reply_text(
        output_string, reply_markup=reply_markup, parse_mode=ParseMode.HTML
    )


async def set_creativity_handler(update: Update, context: CallbackContext) -> None:
    global db

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
    if ulock.locked():
        logger.info("Please wait for your previous request to finish.")
        return

    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        query = update.callback_query
        await query.answer()

        new_creativity = query.data.split("|")[1]
        uprofile = db.get(identifier)
        assert uprofile is not None

        if new_creativity == "zero":
            score = 0.0
        elif new_creativity == "low":
            score = 0.3
        elif new_creativity == "medium":
            score = 0.7
        elif new_creativity == "high":
            score = 1.0
        elif new_creativity == "extreme":
            score = 1.2
        else:
            raise ValueError("Invalid creativity level.")

        uprofile["creativity"] = score
        db.set(identifier, uprofile)

        await context.bot.send_message(
            chat_id=message.chat.id,
            text=f"Creativity set to {new_creativity}.",
            parse_mode=ParseMode.HTML,
        )

    logger.info("Released lock for user: %s", identifier)


async def show_character_menu(update: Update, context: CallbackContext) -> None:
    message: Optional[telegram.Message] = getattr(update, "message", None)
    # edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)

    # is_edit: bool = False
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")

    if message.from_user.id not in PREMIUM_MEMBERS and FREE == "0":
        logger.warning(
            "User (%d |%s) is not a premium member",
            message.from_user.id,
            message.from_user.username,
        )
        await reply(
            message, "You are not a premium member. Contact the author to upgrade."
        )
        return

    characters = []
    for character in CHARACTER.keys():
        if CHARACTER[character]["access"] == "private":
            continue
        characters.append(character)

    output_string = "Click to select a character:\n"

    keyboard = []
    for character in characters:
        name = CHARACTER[character]["name"]
        keyboard.append(
            [
                telegram.InlineKeyboardButton(
                    name, callback_data=f"set_character|{character}"
                )
            ]
        )

    reply_markup = telegram.InlineKeyboardMarkup(keyboard)
    await message.reply_text(
        output_string, reply_markup=reply_markup, parse_mode=ParseMode.HTML
    )


async def set_character_handler(update: Update, context: CallbackContext) -> None:
    global db

    callback_query = update.callback_query

    if callback_query is None:
        raise ValueError("Callback query is None.")

    message = callback_query.message

    if message is None:
        raise ValueError("Message is None.")

    if callback_query.from_user.id not in PREMIUM_MEMBERS and FREE == "0":
        logger.warning(
            "User (%d | %s) is not a premium member.",
            callback_query.from_user.id,
            callback_query.from_user.username,
        )
        await context.bot.send_message(
            chat_id=message.chat.id,
            text="You are not a premium member. Contact the author to upgrade.",
            parse_mode=ParseMode.HTML,
        )
        return

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_user_lock(identifier)
    if ulock.locked():
        logger.info("Please wait for your previous request to finish.")
        return

    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        query = update.callback_query
        await query.answer()

        new_character = query.data.split("|")[1]
        uprofile = db.get(identifier)
        assert uprofile is not None
        uprofile["character"] = new_character
        db.set(identifier, uprofile)

        await context.bot.send_message(
            chat_id=message.chat.id,
            text=f"{CHARACTER[new_character]['welcome_message']}",
            parse_mode=ParseMode.HTML,
        )

    logger.info("Released lock for user: %s", identifier)


async def show_model_menu(update: Update, context: CallbackContext) -> None:
    message: Optional[telegram.Message] = getattr(update, "message", None)
    # edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)

    # is_edit: bool = False
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")

    if message.from_user.id not in PREMIUM_MEMBERS and FREE == "0":
        logger.warning(
            "User (%d |%s) is not a premium member",
            message.from_user.id,
            message.from_user.username,
        )
        await reply(
            message, "You are not a premium member. Contact the author to upgrade."
        )
        return

    output_string = "Click to select a model:\n"

    keyboard = []
    providers = list(PROVIDER.keys())
    for provider in providers:
        models = list(PROVIDER[provider]["t2t"])
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
    global db

    callback_query = update.callback_query

    if callback_query is None:
        raise ValueError("Callback query is None.")

    message = callback_query.message

    if message is None:
        raise ValueError("Message is None.")

    if callback_query.from_user.id not in PREMIUM_MEMBERS and FREE == "0":
        logger.warning(
            "User (%d | %s) is not a premium member.",
            callback_query.from_user.id,
            callback_query.from_user.username,
        )
        await context.bot.send_message(
            chat_id=message.chat.id,
            text="You are not a premium member. Contact the author to upgrade.",
            parse_mode=ParseMode.HTML,
        )
        return

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_user_lock(identifier)
    if ulock.locked():
        logger.info("Please wait for your previous request to finish.")
        return

    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
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


def handle_triple_ticks(text: str, closed: bool):
    TRIPLE_TICKS = "```"
    t = text[:]
    if not closed:
        t = TRIPLE_TICKS + t
    close = len(re.findall(TRIPLE_TICKS, t)) % 2 == 0
    if not close:
        t += TRIPLE_TICKS
    return t, close


def escape_markdown(text) -> str:
    """
    Escape special characters for Telegram's HTMLV2.
    """
    # Characters that need to be escaped
    special_chars = [
        "_",
        # "*",
        "[",
        "]",
        "(",
        ")",
        # "~",
        # "`",
        ">",
        # "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
    ]

    # Escape backslash first to avoid double escaping
    # text = text.replace("\\", "\\\\")
    _text = text[:]
    # # Escape special characters
    for char in special_chars:
        _text = _text.replace(char, f"\\{char}")

    # _text = re.sub(r"#{1,} (.*?)\n", r"*\1*\n", _text)
    _text = re.sub(r"#{2,}", "", _text)
    _text = _text.replace("#", r"\#")

    # logger.info("Raw Markdown: %s", text)
    # logger.info("Escaped Markdown: %s", _text)
    return _text


def escape_html(text) -> str:
    """
    Escape special characters for Telegram's HTMLV2.
    """
    _text = text[:]
    _text = _text.replace("&", "&amp;")
    _text = _text.replace("<", "&lt;")
    _text = _text.replace(">", "&gt;")
    return _text


async def reply(message: telegram.Message, output_string: str) -> None:
    if config.DEBUG == "1":
        logger.info(">> %s", output_string)
    MSG_MAX_LEN = 3800
    if len(output_string) >= MSG_MAX_LEN:
        sections = re.split(r"\n{2,}", output_string)
        current_chunk = ""
        is_close: bool = True
        for section in sections:
            if len(current_chunk) + len(section) + 2 <= MSG_MAX_LEN:
                current_chunk += section + "\n\n"
            else:
                formatted_chunk = escape_markdown(current_chunk)
                formatted_chunk, is_close = handle_triple_ticks(
                    formatted_chunk, is_close
                )
                try:
                    await message.reply_text(
                        output_string, parse_mode=ParseMode.MARKDOWN_V2
                    )
                except telegram.error.BadRequest as bre:
                    logger.error(bre)
                    if "Can't parse entities:" in str(bre):
                        await message.reply_text(
                            escape_html(current_chunk), parse_mode=ParseMode.HTML
                        )
                    else:
                        raise
                current_chunk = section + "\n\n"
        # Handle the last chunk
        if current_chunk:
            formatted_chunk = escape_markdown(current_chunk)
            formatted_chunk, _ = handle_triple_ticks(formatted_chunk, is_close)
            try:
                await message.reply_text(
                    formatted_chunk, parse_mode=ParseMode.MARKDOWN_V2
                )
            except telegram.error.BadRequest as bre:
                logger.error(bre)
                if "Can't parse entities:" in str(bre):
                    await message.reply_text(
                        escape_html(current_chunk), parse_mode=ParseMode.HTML
                    )
                else:
                    raise
    else:
        formatted_chunk, _ = handle_triple_ticks(escape_markdown(output_string), True)
        try:
            await message.reply_text(formatted_chunk, parse_mode=ParseMode.MARKDOWN_V2)
        except telegram.error.BadRequest as bre:
            logger.error(bre)
            if "Can't parse entities:" in str(bre):
                await message.reply_text(
                    escape_html(output_string), parse_mode=ParseMode.HTML
                )
            else:
                raise


async def update_memory(identifier: str, new_content: str | None) -> str | None:
    global agent_zero, chat_memory, db

    umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
    if new_content and umemory is not None:
        if not new_content.startswith("/"):
            umemory.push({"role": "user", "content": new_content})

    uprofile = db.get(identifier)
    if config.DEBUG == "1":
        assert uprofile is not None

    old_interactions = umemory.to_list()[: config.MEMORY_LEN]
    if len(old_interactions) == 0:
        return

    selected_interactions = []
    for interaction in old_interactions:
        if interaction["role"] == "user":
            selected_interactions.append(interaction)

    context = selected_interactions if len(selected_interactions) > 0 else None
    existing_memory = uprofile["memory"]
    if existing_memory:
        prompt = f"Update user's metadata. Existing metadata: \n{existing_memory}"
    else:
        prompt = "Construct user's metadata."

    responses = await agent_zero.run_async(
        query=prompt, context=context, mode=ResponseMode.JSON
    )
    uprofile["memory"] = responses[0]["content"]
    db.set(identifier, uprofile)


async def middleware_function(update: Update, context: CallbackContext) -> None:
    """
    Intercepts messages
    """
    global chat_memory, rate_limiter, db, user_stats

    logger.info("Middleware => Update: %s", update)
    message: Optional[telegram.Message] = getattr(update, "message", None)
    # edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)

    # is_edit: bool = False
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")
        # is_edit = True
    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_user_lock(identifier)
    if ulock.locked():
        await message.reply_text("Please wait for your previous request to finish.")

    async with ulock:
        allowed_to_pass = rate_limiter.check_limit(identifier)
        user_stats[identifier] = allowed_to_pass
        if not allowed_to_pass:
            return

        register_user(identifier)
        register_memory(identifier)
        await update_memory(identifier, message.text)


async def help_handler(update: Update, context: CallbackContext) -> None:
    repo_path: str = config.REPO_PATH

    message: Optional[telegram.Message] = getattr(update, "message", None)
    # edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)

    # is_edit: bool = False
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")

    await message.reply_text(f"GitHub: {repo_path}")


async def start_handler(update: Update, context: CallbackContext) -> None:
    repo_path: str = config.REPO_PATH

    message: Optional[telegram.Message] = getattr(update, "message", None)
    # edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)

    # is_edit: bool = False
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")

    await message.reply_text(
        f"Thank you for trying out my project. You can find me at GitHub: {repo_path}"
    )


async def error_handler(update: object, context: CallbackContext):
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    logger.info("\nError Handler => Update: %s", update)


async def call_llm(llm, prompt, context, mode, response_format) -> list[dict]:
    MAX_RETRY = 5
    iteration = 0
    while iteration < MAX_RETRY:
        try:
            responses = await llm.run_async(
                query=prompt, context=context, mode=mode, format=response_format
            )
            return responses
        except ValueError as ve:
            if "max_output_tokens <= 0" in str(ve):
                context = context[1:]
                iteration += 1
            else:
                logger.error("call_llm: ValueError: %s", ve)
                raise
        except Exception as e:
            logger.error("call_llm: Exception: %s", e)
            raise
    raise ValueError(f"max_output_tokens <= 0. Retry up to {MAX_RETRY} times.")


async def call_ii(
    ii: ImageInterpreter, prompt: str, context: list[dict] | None, filepath: str
) -> list[dict]:
    MAX_RETRY = 5
    iteration = 0
    while iteration < MAX_RETRY:
        try:
            responses = await ii.interpret_async(
                query=prompt,
                context=context,
                filepath=filepath,
                mode=ResponseMode.SO,
                format=IIResponse,
            )
            return responses
        except ValueError as ve:
            if str(ve) == "max_output_tokens <= 0":
                context = context[1:]
                iteration += 1
            else:
                raise
    raise ValueError(f"max_output_tokens <= 0. Retry up to {MAX_RETRY} times.")


async def message_handler(update: Update, context: CallbackContext) -> None:
    global chat_memory, user_locks, db, user_stats

    message: Optional[telegram.Message] = getattr(update, "message", None)
    # edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)

    # is_edit: bool = False
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_user_lock(identifier)
    if ulock.locked():
        logger.info("Please wait for your previous request to finish.")
        return

    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, "Exceeded rate limit. Please try again later.")
            logger.info("Released lock for user: %s", identifier)
            return

        prompt: str = message.text
        if prompt is None or prompt == "":
            raise ValueError("Prompt is None or empty.")

        umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
        if umemory is None:
            raise ValueError("Memory is None.")

        recent_conversation = umemory.last_n(n=MEMORY_LEN)

        # flist = umemory.to_list()
        # for m in flist:
        #     logger.info("Memory: %s", m)

        # logger.info("Recent: %s", recent_conversation)

        if len(recent_conversation) > 1:
            recent_conversation = recent_conversation[:-1]  # The last one is the prompt
        else:
            recent_conversation = None

        uprofile = db.get(identifier)
        assert uprofile is not None

        platform = uprofile["platform_t2t"]
        model_name = uprofile["model_t2t"]
        character = uprofile["character"]
        umetadata = uprofile["memory"]
        if umetadata:
            if recent_conversation:
                recent_conversation.insert(
                    0,
                    {"role": "system", "content": f"<metadata>{umetadata}</metadata>"},
                )

        auto_routing = uprofile["auto_routing"]
        if auto_routing:
            if recent_conversation and len(recent_conversation) >= 3:
                character = await find_best_agent(
                    prompt, context=recent_conversation[-3:]
                )
            else:
                character = await find_best_agent(prompt, context=None)

        llm = llm_factory.create_chat_llm(
            platform, model_name, character, uprofile["creativity"]
        )
        # logger.info("System Prompt: %s", llm.system_prompt)
        responses = await call_llm(
            llm, prompt, recent_conversation, ResponseMode.DEFAULT, None
        )
        logger.info("Generated %d responses.", len(responses))
        for response in responses:
            logger.info("\nResponse: %s", response["content"])

        response = responses[-1]
        content = response["content"]
        # logger.info("Raw Response: %s", content)

        # jresult = json.loads(content)
        # if "result" not in jresult:
        #     output_string = "Sorry. Please try again."
        #     await reply(message, output_string)
        #     return

        # for key in jresult.keys():
        #     logger.info("%s: %s", key, jresult[key])

        umemory.push({"role": "assistant", "content": content})

        # if CONFIG.DEBUG == "1":
        #     # View how the llm handle user's prompt
        #     for idx, step in enumerate(jresult["steps"], start=1):
        #         progress = f"[{idx}] Goal={step['goal']}\nTask={step['task']}\nOutput={step['output']}\n\n"
        #         logger.info(progress)

        output_string = CHARACTER[character]["name"] + ":\n" + content

        if output_string is None or output_string == "":
            output_string = "Sorry."

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
    # edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)

    # is_edit: bool = False
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")
    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_user_lock(identifier)
    if ulock.locked():
        logger.info("Please wait for your previous request to finish.")
        return

    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        allowed_to_pass = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, "Exceeded rate limit. Please try again later.")
            logger.info("Released lock for user: %s", identifier)
            return

        user_folder = f"/temp/{identifier}"
        if not os.path.exists(user_folder):
            os.mkdir(user_folder)
        photo = message.photo[-1]
        file_id: str = photo.file_id

        photo_file = await context.bot.get_file(file_id)
        export_path = os.path.join(
            user_folder, generate_unique_filename(file_id, "jpg", True)
        )
        await photo_file.download_to_drive(
            export_path,
            read_timeout=3000,
            write_timeout=3000,
            connect_timeout=3000,
        )

        umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
        if umemory is None:
            raise ValueError("Memory is None.")

        prompt = "Describe the image."
        if message.caption:
            prompt += f" Caption={message.caption}"

        umemory.push({"role": "user", "content": prompt})

        # recent_conversation = umemory.last_n(n=MEMORY_LEN)
        uprofile = db.get(identifier)
        assert uprofile is not None

        platform = uprofile["platform_i2t"]
        model_name = uprofile["model_i2t"]
        system_prompt = CHARACTER["seer"]["system_prompt"]
        image_interpreter = llm_factory.create_image_interpreter(
            platform, model_name, system_prompt, uprofile["creativity"]
        )
        responses = await call_ii(image_interpreter, prompt, None, filepath=export_path)
        response = responses[-1]

        content_string = response["content"]
        umemory.push({"role": "assistant", "content": content_string})

        jresult = json.loads(content_string)

        output_string = ""
        # summary
        if "summary" in jresult:
            summary = jresult["summary"]
            output_string += f"**Summary**\n{summary}\n\n"

        # long_description
        if "long_description" in jresult:
            long_description = jresult["long_description"]
            output_string += f"**Description**\n{long_description}\n\n"

        # keywords
        if "keywords" in jresult:
            keywords = jresult["keywords"]
            if not isinstance(keywords, list):
                logger.warning(
                    "Encounter invalid response schema. Expect keywords to be a list of str, but get '%s'.",
                    type(keywords).__name__,
                )
            else:
                kws = ", ".join(keywords)
                output_string += f"**Keywords**\n[{kws}]"

        if output_string == "":
            raise RuntimeError("Content String is empty.")
        await reply(message, output_string)

    logger.info("Released lock for user: %s", identifier)


async def reset_chatmemory_handler(update: Update, context: CallbackContext):
    global chat_memory, user_locks

    message: Optional[telegram.Message] = getattr(update, "message", None)
    # edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)

    # is_edit: bool = False
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_user_lock(identifier)
    if ulock.locked():
        logger.info("Please wait for your previous request to finish.")
        return

    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        umemory = chat_memory.get(identifier, None)
        if umemory is None:
            raise ValueError("Memory is None.")

        umemory.clear()
        await message.reply_text("Memory has been reset.")

    logger.info("Released lock for user: %s", identifier)


async def reset_user_handler(update: Update, context: CallbackContext):
    global user_locks

    message: Optional[telegram.Message] = getattr(update, "message", None)
    # edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)

    # is_edit: bool = False
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_user_lock(identifier)
    if ulock.locked():
        logger.info("Please wait for your previous request to finish.")
        return

    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
        register_memory(identifier, force=True)
        register_user(identifier)
        await message.reply_text("User has been reset.")
    logger.info("Released lock for user: %s", identifier)


class CompressContent(BaseModel):
    topics: list[str]
    compressed: str


async def compress_memory_handler(update: Update, context: CallbackContext):
    global chat_memory, user_locks, db

    message: Optional[telegram.Message] = getattr(update, "message", None)
    # edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)

    # is_edit: bool = False
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    ulock = get_user_lock(identifier)
    if ulock.locked():
        logger.info("Please wait for your previous request to finish.")
        return

    async with ulock:
        logger.info("Acquired lock for user: %s", identifier)
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
        # system_prompt = CHARACTER["general"]["system_prompt"]
        llm = llm_factory.create_chat_llm(platform, model_name, "general", 0.0)

        responses = await call_llm(llm, prompt, context, ResponseMode.DEFAULT, None)

        response = responses[0]
        content = response["content"]

        if content is None or content == "":
            raise ValueError("Content is None or empty.")

        # logger.info("Compressed: %s", content)
        # _ = json.loads(content)

        umemory.clear()
        umemory.push({"role": "assistant", "content": content})
        # chat_memory[identifier] = umemory

        await message.reply_text("Memory has been compressed.")

    logger.info("Released lock for user: %s", identifier)
