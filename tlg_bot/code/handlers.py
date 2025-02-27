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

# External Imports
import telegram
from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode
from pydantic import BaseModel

from llm_agent_toolkit import ResponseMode, ShortTermMemory  # type: ignore
from llm_agent_toolkit._core import ImageInterpreter
import chromadb

from llms import LLMFactory, IIResponse
from pygrl import BasicStorage, GeneralRateLimiter as grl

# Internal Imports
import storage
import custom_library
import custom_workflow
from transcriber import TranscriberFactory
import config

logger = logging.getLogger(__name__)

# Global Variables
chat_memory: dict[str, ShortTermMemory] = {}
user_locks: dict[str, Lock] = {}
db = storage.SQLite3_Storage("/db/ost.db", "user_profile", False)

rl_storage = BasicStorage()
rate_limiter = grl(rl_storage, 1, 1, 100)
user_stats: dict[str, tuple[bool, str]] = {}

main_vdb: chromadb.ClientAPI = custom_library.ChromaDBFactory.get_instance(
    persist=True, persist_directory="/temp/vect"
)
web_db = storage.WebCache(ttl=600, maxsize=128)
llm_factory = LLMFactory(vdb=main_vdb, webcache=web_db)

agent_zero = llm_factory.create_chat_llm(
    "openai", "gpt-4o-mini", "extractor", 0.3, False
)
agent_router = llm_factory.create_chat_llm("openai", "gpt-4o-mini", "router", 0.3, True)
transcriber_factory = TranscriberFactory(
    provider="local", model_name="turbo", output_directory="/temp", audio_parameter=None
)


def register_user(identifier: str, force: bool = False) -> None:
    global db
    uprofile = db.get(identifier)
    if uprofile is None or force:
        pt2t, mt2t = config.DEFAULT_T2T_MODEL
        pi2t, mi2t = config.DEFAULT_I2T_MODEL
        db.set(
            identifier,
            {
                "platform_t2t": pt2t,
                "model_t2t": mt2t,
                "platform_i2t": pi2t,
                "model_i2t": mi2t,
                "character": config.DEFAULT_CHARACTER,
                "creativity": 0.7,
                "memory": None,
                "auto_routing": True,
            },
        )


def register_memory(identifier: str, force: bool = False) -> None:
    global chat_memory

    umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
    if umemory is None or force:
        chat_memory[identifier] = ShortTermMemory(max_entry=config.MEMORY_LEN + 5)


async def show_character_handler(update: Update, context: CallbackContext) -> None:
    """
    TODO: ONLY show MEMORY!!!
    """
    global db, user_stats
    message: Optional[telegram.Message] = getattr(update, "message", None)
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
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        uprofile = db.get(identifier)
        assert uprofile is not None

        character = uprofile["character"]
        character_name = config.CHARACTER[character]["name"]
        system_prompt = config.CHARACTER[character]["system_prompt"]
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
            output_string = "*Memory*:\n```\n" + memory + "\n```"
        else:
            output_string = "It takes a while to build user profile. "

        await reply(message, output_string)

    logger.info("Released lock for user: %s", identifier)


# async def show_creativity_menu(update: Update, context: CallbackContext) -> None:
#     message: Optional[telegram.Message] = getattr(update, "message", None)
#     if message is None:
#         message = getattr(update, "edited_message", None)
#         if message is None:
#             raise ValueError("Message is None.")

#     output_string = "Click to select a level:\n"

#     keyboard = []
#     for level in ["zero", "low", "medium", "high", "extreme"]:
#         keyboard.append(
#             [
#                 telegram.InlineKeyboardButton(
#                     level, callback_data=f"set_creativity|{level}"
#                 )
#             ]
#         )

#     reply_markup = telegram.InlineKeyboardMarkup(keyboard)
#     await message.reply_text(
#         output_string, reply_markup=reply_markup, parse_mode=ParseMode.HTML
#     )


# async def set_creativity_handler(update: Update, context: CallbackContext) -> None:
#     global db

#     callback_query = update.callback_query

#     if callback_query is None:
#         raise ValueError("Callback query is None.")

#     message = callback_query.message

#     if message is None:
#         raise ValueError("Message is None.")

#     identifier: str = (
#         f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
#     )

#     ulock = get_user_lock(identifier)
#     if ulock.locked():
#         logger.info("Please wait for your previous request to finish.")
#         return

#     async with ulock:
#         logger.info("Acquired lock for user: %s", identifier)
#         query = update.callback_query
#         await query.answer()

#         new_creativity = query.data.split("|")[1]
#         uprofile = db.get(identifier)
#         assert uprofile is not None

#         if new_creativity == "zero":
#             score = 0.0
#         elif new_creativity == "low":
#             score = 0.3
#         elif new_creativity == "medium":
#             score = 0.7
#         elif new_creativity == "high":
#             score = 1.0
#         elif new_creativity == "extreme":
#             score = 1.2
#         else:
#             raise ValueError("Invalid creativity level.")

#         uprofile["creativity"] = score
#         db.set(identifier, uprofile)

#         await context.bot.send_message(
#             chat_id=message.chat.id,
#             text=f"Creativity set to {new_creativity}.",
#             parse_mode=ParseMode.HTML,
#         )

#     logger.info("Released lock for user: %s", identifier)


# async def show_character_menu(update: Update, context: CallbackContext) -> None:
#     message: Optional[telegram.Message] = getattr(update, "message", None)
#     if message is None:
#         message = getattr(update, "edited_message", None)
#         if message is None:
#             raise ValueError("Message is None.")

#     if message.from_user.id not in config.PREMIUM_MEMBERS and config.FREE == "0":
#         logger.warning(
#             "User (%d |%s) is not a premium member",
#             message.from_user.id,
#             message.from_user.username,
#         )
#         await reply(
#             message, "You are not a premium member. Contact the author to upgrade."
#         )
#         return

#     characters = []
#     for character in config.CHARACTER.keys():
#         if config.CHARACTER[character]["access"] == "private":
#             continue
#         characters.append(character)

#     output_string = "Click to select a character:\n"

#     keyboard = []
#     for character in characters:
#         name = config.CHARACTER[character]["name"]
#         keyboard.append(
#             [
#                 telegram.InlineKeyboardButton(
#                     name, callback_data=f"set_character|{character}"
#                 )
#             ]
#         )

#     reply_markup = telegram.InlineKeyboardMarkup(keyboard)
#     await message.reply_text(
#         output_string, reply_markup=reply_markup, parse_mode=ParseMode.HTML
#     )


# async def set_character_handler(update: Update, context: CallbackContext) -> None:
#     global db, user_stats

#     callback_query = update.callback_query

#     if callback_query is None:
#         raise ValueError("Callback query is None.")

#     message = callback_query.message

#     if message is None:
#         raise ValueError("Message is None.")

#     # if callback_query.from_user.id not in config.PREMIUM_MEMBERS and config.FREE == "0":
#     #     logger.warning(
#     #         "User (%d | %s) is not a premium member.",
#     #         callback_query.from_user.id,
#     #         callback_query.from_user.username,
#     #     )
#     #     await context.bot.send_message(
#     #         chat_id=message.chat.id,
#     #         text="You are not a premium member. Contact the author to upgrade.",
#     #         parse_mode=ParseMode.HTML,
#     #     )
#     #     return

#     identifier: str = (
#         f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
#     )

#     ulock = get_user_lock(identifier)
#     if ulock.locked():
#         logger.info("Please wait for your previous request to finish.")
#         return

#     async with ulock:
#         logger.info("Acquired lock for user: %s", identifier)
#         allowed_to_pass, _msg = user_stats[identifier]
#         if not allowed_to_pass:
#             await reply(message, _msg)
#             logger.info("Released lock for user: %s", identifier)
#             return

#         query = update.callback_query
#         await query.answer()

#         new_character = query.data.split("|")[1]
#         uprofile = db.get(identifier)
#         assert uprofile is not None
#         uprofile["character"] = new_character
#         db.set(identifier, uprofile)

#         await context.bot.send_message(
#             chat_id=message.chat.id,
#             text=f"{config.CHARACTER[new_character]['welcome_message']}",
#             parse_mode=ParseMode.HTML,
#         )

#     logger.info("Released lock for user: %s", identifier)


async def show_model_menu(update: Update, context: CallbackContext) -> None:
    global user_stats

    message: Optional[telegram.Message] = getattr(update, "message", None)
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
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        output_string = "Click to select a model:\n"

        keyboard = []
        providers = list(config.PROVIDER.keys())
        for provider in providers:
            models = list(config.PROVIDER[provider]["t2t"])
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
    if ulock.locked():
        logger.info("Please wait for your previous request to finish.")
        return

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
                formatted_chunk = custom_library.escape_markdown(current_chunk)
                formatted_chunk, is_close = custom_library.handle_triple_ticks(
                    formatted_chunk, is_close
                )
                try:
                    await message.reply_text(
                        formatted_chunk, parse_mode=ParseMode.MARKDOWN_V2
                    )
                except telegram.error.BadRequest as bre:
                    logger.error(bre)
                    if "Can't parse entities:" in str(bre):
                        await message.reply_text(
                            custom_library.escape_html(current_chunk),
                            parse_mode=ParseMode.HTML,
                        )
                    else:
                        raise
                current_chunk = section + "\n\n"
        # Handle the last chunk
        if current_chunk:
            formatted_chunk = custom_library.escape_markdown(current_chunk)
            formatted_chunk, _ = custom_library.handle_triple_ticks(
                formatted_chunk, is_close
            )
            try:
                await message.reply_text(
                    formatted_chunk, parse_mode=ParseMode.MARKDOWN_V2
                )
            except telegram.error.BadRequest as bre:
                logger.error(bre)
                if "Can't parse entities:" in str(bre):
                    await message.reply_text(
                        custom_library.escape_html(current_chunk),
                        parse_mode=ParseMode.HTML,
                    )
                else:
                    raise
    else:
        formatted_chunk, _ = custom_library.handle_triple_ticks(
            custom_library.escape_markdown(output_string), True
        )
        try:
            await message.reply_text(formatted_chunk, parse_mode=ParseMode.MARKDOWN_V2)
        except telegram.error.BadRequest as bre:
            logger.error(bre)
            if "Can't parse entities:" in str(bre):
                await message.reply_text(
                    custom_library.escape_html(output_string), parse_mode=ParseMode.HTML
                )
            else:
                raise


async def middleware_function(update: Update, context: CallbackContext) -> None:
    """
    Intercepts messages
    """
    global chat_memory, rate_limiter, db, user_stats, agent_zero

    if config.DEBUG == "1":
        logger.info("Middleware => Update: %s", update)
    message: Optional[telegram.Message] = getattr(update, "message", None)
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")
        # is_edit = True

    identifier: str = (
        f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    )

    if message.chat.id not in config.PREMIUM_MEMBERS:
        logger.warning("Unauthorized Access: %d", message.chat.id)
        user_stats[identifier] = (False, "Unauthorized Access")
        return

    ulock = get_user_lock(identifier)
    if ulock.locked():
        await message.reply_text("Please wait for your previous request to finish.")

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
        if message.text:
            await custom_workflow.update_memory(
                agent_zero, chat_memory, db, identifier, message.text
            )

        if message.voice or message.audio:
            if message.voice:
                file_id = message.voice.file_id
                file_extension = message.voice.mime_type.split("/")[-1]
            else:
                file_id = message.audio.file_id
                file_extension = message.audio.mime_type.split("/")[-1]

            if file_extension == "mpeg":
                file_extension = "mp3"

            user_folder = f"/temp/{identifier}"
            temp_path = os.path.join(user_folder, f"{file_id}.{file_extension}")
            saved: bool = await store_to_drive(file_id, temp_path, context)
            if saved:
                audio_agent = transcriber_factory.get_transcriber()
                user_folder = f"/temp/{identifier}"
                responses = await audio_agent.transcribe_async(
                    prompt="voice input",
                    filepath=temp_path,
                    tmp_directory=user_folder,
                )
                transcript = responses[0]["content"]
                transcript_json = json.loads(transcript)
                content = ""
                for t in transcript_json["transcript"]:
                    content += t["text"] + "\n"

                if message.voice:
                    await reply(message, f"**Whisper 🎤**:\n{content}")
                    await custom_workflow.update_memory(
                        agent_zero,
                        chat_memory,
                        db,
                        identifier,
                        f"Voice Input:{content}",
                    )
                elif message.audio:
                    filepostfix = datetime.now().isoformat()
                    filepostfix = filepostfix.replace(":", "-")
                    filepostfix = filepostfix.split(".")[0]

                    filename = f"audio_{filepostfix}.txt"
                    content_txt_path = os.path.join(user_folder, filename)
                    with open(content_txt_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    await message.reply_document(
                        document=content_txt_path,
                        caption=message.caption,
                        allow_sending_without_reply=True,
                        filename=filename,
                    )
                    await custom_workflow.update_memory(
                        agent_zero,
                        chat_memory,
                        db,
                        identifier,
                        f"Audio Upload:{content}",
                    )
                # Remove sliced audio files
                temp_files = list(
                    filter(lambda x: x.startswith("slice"), os.listdir(user_folder))
                )
                for tfile in temp_files:
                    tfile_path = os.path.join(f"{user_folder}/{tfile}")
                    os.remove(tfile_path)


async def store_to_drive(file_id: str, temp_path: str, context: CallbackContext):
    from telegram.error import TelegramError

    if os.path.exists(temp_path):
        return False
    try:
        _file = await context.bot.get_file(
            file_id,
            read_timeout=300,
            write_timeout=300,
            connect_timeout=300,
            pool_timeout=300,
        )
        await _file.download_to_drive(
            temp_path,
            read_timeout=3000,
            write_timeout=3000,
            connect_timeout=3000,
            pool_timeout=300,
        )
        return True
    except TelegramError as tg_err:
        logger.error("[store_to_drive]=TelegramError: %s", str(tg_err))
    except Exception as e:
        logger.error("[store_to_drive]=Exception: %s", str(e))
    # raise RuntimeError(f"({file_id}, {temp_path}) => File download failed.")
    return False


async def help_handler(update: Update, context: CallbackContext) -> None:
    repo_path: str = config.REPO_PATH

    message: Optional[telegram.Message] = getattr(update, "message", None)

    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")

    await message.reply_text(
        f"Thank you for trying out my project. You can find me at GitHub: {repo_path}"
    )


async def start_handler(update: Update, context: CallbackContext) -> None:
    repo_path: str = config.REPO_PATH

    message: Optional[telegram.Message] = getattr(update, "message", None)

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
        logger.info("Attempt: %d", iteration)
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
                iteration += 1
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
            if context and str(ve) == "max_output_tokens <= 0":
                context = context[1:]
                iteration += 1
            else:
                raise
    raise ValueError(f"max_output_tokens <= 0. Retry up to {MAX_RETRY} times.")


async def call_chat_llm(
    profile: dict, memory: ShortTermMemory, prompt: str
) -> tuple[str, ShortTermMemory]:
    platform = profile["platform_t2t"]
    model_name = profile["model_t2t"]
    character = profile["character"]
    metadata = profile["memory"]

    recent_conv = memory.last_n(config.MEMORY_LEN)
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
        character = await custom_workflow.find_best_agent(
            agent_router,
            prompt,
            context=recent_conv[-3:] if recent_conv else recent_conv,
        )

    llm = llm_factory.create_chat_llm(
        platform, model_name, character, profile["creativity"]
    )
    responses = await call_llm(llm, prompt, recent_conv, ResponseMode.DEFAULT, None)

    logger.info("Generated %d responses.", len(responses))
    for response in responses:
        # logger.info("\nResponse: %s", response["content"])
        memory.push({"role": "assistant", "content": response["content"]})

    final_response = responses[-1]
    output_string = (
        config.CHARACTER[character]["name"] + ":\n" + final_response["content"]
    )
    return output_string, memory


async def message_handler(update: Update, context: CallbackContext) -> None:
    global chat_memory, user_locks, db, user_stats, agent_router

    message: Optional[telegram.Message] = getattr(update, "message", None)
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
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        prompt: str = message.text
        if prompt is None or prompt == "":
            raise ValueError("Prompt is None or empty.")

        umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
        if umemory is None:
            raise ValueError("Memory is None.")

        uprofile = db.get(identifier)
        assert uprofile is not None

        output_string, updated_memory = await call_chat_llm(uprofile, umemory, prompt)
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
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        user_folder = f"/temp/{identifier}"
        if not os.path.exists(user_folder):
            os.mkdir(user_folder)

        photo = message.photo[-1]

        file_id: str = photo.file_id
        export_path = os.path.join(
            user_folder, generate_unique_filename(file_id, "jpg", True)
        )
        saved: bool = await store_to_drive(file_id, export_path, context)
        if not saved and not os.path.exists(export_path):
            logger.warning("Failed to saved %s to \n%s.", file_id, export_path)
            await reply(message, "Fail to process upload file. Please try again.")
            return None

        umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
        if umemory is None:
            raise ValueError("Memory is None.")

        prompt = "Describe the image."
        if message.caption:
            prompt += f" Caption={message.caption}"

        umemory.push({"role": "user", "content": prompt})

        uprofile = db.get(identifier)
        assert uprofile is not None

        platform = uprofile["platform_i2t"]
        model_name = uprofile["model_i2t"]
        system_prompt = config.CHARACTER["seer"]["system_prompt"]
        image_interpreter = llm_factory.create_image_interpreter(
            platform, model_name, system_prompt, uprofile["creativity"]
        )
        responses = await call_ii(image_interpreter, prompt, None, filepath=export_path)
        response = responses[-1]

        content_string = response["content"]

        jresult = json.loads(content_string)

        output_string = f"Image Upload, interpreted by {model_name}:\n"
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

        if output_string == f"Image Upload, interpreted by {model_name}:\n":
            logger.warning(
                "Expected fields not found in the response: %s", content_string
            )
            output_string += jresult["text"]

        umemory.push({"role": "user", "content": output_string})

        await reply(message, output_string)

        prompt = f"{output_string}"
        if message.caption:
            prompt += f"\nCaption={message.caption}"

        prompt += "\nKeep it with you. Don't have to response/answer/comment. Chill!"
        output_string, updated_memory = await call_chat_llm(uprofile, umemory, prompt)
        updated_memory.push({"role": "assistant", "content": output_string})
        chat_memory[identifier] = updated_memory
        await reply(message, output_string)

    logger.info("Released lock for user: %s", identifier)


async def reset_chatmemory_handler(update: Update, context: CallbackContext):
    global chat_memory, user_locks, user_stats

    message: Optional[telegram.Message] = getattr(update, "message", None)
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

        responses = await call_llm(llm, prompt, context, ResponseMode.DEFAULT, None)

        response = responses[0]
        content = response["content"]

        if content is None or content == "":
            raise ValueError("Content is None or empty.")

        umemory.clear()
        umemory.push({"role": "assistant", "content": content})

        await message.reply_text("Memory has been compressed.")

    logger.info("Released lock for user: %s", identifier)


async def voice_handler(update: Update, context: CallbackContext) -> None:
    global chat_memory, user_locks, db, user_stats, agent_router

    logger.info("The VOICE!!!")
    message: Optional[telegram.Message] = getattr(update, "message", None)
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
        allowed_to_pass, _msg = user_stats[identifier]
        if not allowed_to_pass:
            await reply(message, _msg)
            logger.info("Released lock for user: %s", identifier)
            return

        umemory: Optional[ShortTermMemory] = chat_memory.get(identifier, None)
        if umemory is None:
            raise ValueError("Memory is None.")

        recent_conversation = umemory.last_n(n=config.MEMORY_LEN)
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
                character = await custom_workflow.find_best_agent(
                    agent_router, prompt, context=recent_conversation[-3:]
                )
            else:
                character = await custom_workflow.find_best_agent(
                    agent_router, prompt, context=None
                )

        logger.info("Model Name=%s, Character=%s", model_name, character)
        llm = llm_factory.create_chat_llm(
            platform, model_name, character, uprofile["creativity"]
        )
        logger.info("llm=%s", llm)
        responses = await call_llm(
            llm, prompt, recent_conversation, ResponseMode.DEFAULT, None
        )
        logger.info("Generated %d responses.", len(responses))
        for response in responses:
            logger.info("\nResponse: %s", response["content"])

        response = responses[-1]
        content = response["content"]

        umemory.push({"role": "assistant", "content": content})

        output_string = config.CHARACTER[character]["name"] + ":\n" + content

        if output_string is None or output_string == "":
            output_string = "Sorry."

        await reply(message, output_string)

    logger.info("Released lock for user: %s", identifier)


async def audio_handler(update: Update, context: CallbackContext) -> None:
    global chat_memory, user_locks, db, user_stats, agent_router

    logger.info("The AUDIO!!!")
    message: Optional[telegram.Message] = getattr(update, "message", None)
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

    await reply(message, "COPY")
