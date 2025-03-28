import os
import json
import re
import logging
from typing import Any, Optional
from datetime import datetime
import telegram
from telegram import Message
from telegram.ext import CallbackContext
from telegram.error import TelegramError
from telegram.constants import ParseMode
from llm_agent_toolkit import Core, ResponseMode, ShortTermMemory, ImageInterpreter
from llm_agent_toolkit._util import TokenUsage
from llm_agent_toolkit.transcriber import Transcriber

from custom_library import store_to_drive, unpack_ii_content
from llms import LLMFactory, IIResponse

import mystorage
from myconfig import CHARACTER, DEFAULT_CHARACTER, MEMORY_LEN, DEBUG

logger = logging.getLogger(__name__)


async def find_best_agent(
    agent_router: Core, prompt: str, context: list[dict] | None
) -> tuple[str, TokenUsage]:
    # Input
    agents = []
    for character in CHARACTER.keys():
        if CHARACTER[character]["io"] == "i2t":
            continue

        if CHARACTER[character]["access"] == "private":
            continue

        agent_object = {
            "name": character,
            "profile": CHARACTER[character]["profile"],
        }
        t = CHARACTER[character].get("tools", None)
        # heavily rely on the tool name, no further description here.
        if t:
            agent_object["tools"] = t
        agents.append(agent_object)

    input_prompt = {"request": prompt, "agents": agents}
    if context:
        input_prompt["context"] = context

    # This step makes the prompt unnecessarily long
    # TODO: Optimization needed
    templated_prompt = json.dumps(input_prompt, ensure_ascii=False)
    logger.warning("Templated Prompt LENGTH: %d", len(templated_prompt))
    responses, usage = await agent_router.run_async(
        query=templated_prompt, context=None, mode=ResponseMode.JSON
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
        logger.warning(
            "[find_best_agent]\nPick %s. \nReason: %s\nToken Usage: %s",
            best_agent,
            reason,
            usage,
        )
        return best_agent, usage
    except json.JSONDecodeError:
        return DEFAULT_CHARACTER, usage


async def update_preference(
    agent: Core,
    umemory: ShortTermMemory,
    existing_preference: str | None = None,
    # db: mystorage.SQLite3_Storage,
    # identifier: str,
    # new_content: str | None,
) -> tuple[str | None, TokenUsage]:
    old_interactions = umemory.to_list()[:MEMORY_LEN]
    if len(old_interactions) == 0:
        return existing_preference, TokenUsage(input_tokens=0, output_tokens=0)

    selected_interactions = []
    for interaction in old_interactions:
        if interaction["role"] == "user":
            selected_interactions.append(
                {"type": "query", "content": interaction["content"]}
            )
        else:
            selected_interactions.append(
                {"type": "response", "content": interaction["content"]}
            )

    # context = selected_interactions if len(selected_interactions) > 0 else None
    # existing_memory = uprofile["memory"]
    instructions = []
    if existing_preference:
        instructions.extend(
            [
                "Instruction:\nUpdate user's metadata.",
                f"Existing Metadata: {existing_preference}",
            ]
        )
    else:
        instructions.append("Construct user's metadata.")

    if len(selected_interactions) > 0:
        # This step makes the prompt unnecessarily long
        # TODO: Optimization needed
        instructions.append(
            f"Recent Interactions: {json.dumps(selected_interactions, ensure_ascii=False)}"
        )

    prompt = "\n\n".join(instructions)
    responses, usage = await agent.run_async(
        query=prompt, context=None, mode=ResponseMode.JSON
    )
    logger.info(
        "[update_preference]\nResponses: %d\nToken Usage: %s", len(responses), usage
    )
    # uprofile["memory"] = responses[-1]["content"]
    # db.set(identifier, uprofile)
    return responses[-1]["content"], usage


async def process_audio_input(
    message: Message,
    context: CallbackContext,
    transcriber: Transcriber,
    user_folder: str,
):
    if message.voice:
        file_id = message.voice.file_id
        file_extension = message.voice.mime_type.split("/")[-1]
    else:
        file_id = message.audio.file_id
        file_extension = message.audio.mime_type.split("/")[-1]

    if file_extension == "mpeg":
        file_extension = "mp3"

    temp_path = os.path.join(user_folder, f"{file_id}.{file_extension}")
    try:
        await store_to_drive(file_id, temp_path, context, False)
    except TelegramError as te:
        if str(te) == "File is too big":
            await message.reply_text(
                "<b>Warning</b>: File is too big.", parse_mode=ParseMode.HTML
            )
        return None, None
    except Exception:
        return None, None

    try:
        responses = await transcriber.transcribe_async(
            prompt="voice input",
            filepath=temp_path,
            tmp_directory=user_folder,
        )
        transcript = responses[-1]["content"]
        transcript_json = json.loads(transcript)
        content = ""
        for t in transcript_json["transcript"]:
            content += t["text"] + "\n"
    except Exception as e:
        logger.error("transcribe_async: %s", str(e), exc_info=True, stack_info=True)
        raise
    finally:
        # Remove sliced audio files
        temp_files = list(
            filter(lambda x: x.startswith("slice"), os.listdir(user_folder))
        )
        for tfile in temp_files:
            os.remove(os.path.join(f"{user_folder}/{tfile}"))

    content_txt_path = None
    if message.voice:
        output_string = content
    else:  # audio
        filepostfix = datetime.now().isoformat()
        filepostfix = filepostfix.replace(":", "-")
        filepostfix = filepostfix.split(".")[0]

        filename = f"transcript_{filepostfix}.txt"
        content_txt_path = os.path.join(user_folder, filename)
        with open(content_txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        # await message.reply_document(
        #     document=content_txt_path,
        #     caption=message.caption,
        #     allow_sending_without_reply=True,
        #     filename=filename,
        # )
        output_string = content

    return output_string, content_txt_path


async def call_ii(
    ii: ImageInterpreter, prompt: str, context: list[dict] | None, filepath: str
) -> tuple[list[dict], TokenUsage]:
    MAX_RETRY = 3
    iteration = 0
    while iteration < MAX_RETRY:
        try:
            responses, usage = await ii.interpret_async(
                query=prompt,
                context=context,
                filepath=filepath,
                mode=ResponseMode.SO,
                format=IIResponse,
            )
            return responses, usage
        except ValueError as ve:
            if str(ve) != "max_output_tokens <= 0":
                raise
            logger.warning("call_ii: max_output_tokens <= 0")
            if context is None or len(context) == 0:
                raise
            logger.warning("Reduce context length and try again.")
            context = context[1:]
            iteration += 1
        except Exception as e:
            logger.error("call_ii: Exception: %s", e)
            break
    raise ValueError(f"call_ii FAILED. Tried {iteration} times.")


async def image_interpreter_pipeline(
    profile: dict, prompt: str, filepath: str, llm_factory: LLMFactory
) -> tuple[str, dict]:
    logger.info("Execute image_interpreter_pipeline")
    platform = profile["platform_i2t"]
    model_name = profile["model_i2t"]
    system_prompt = CHARACTER["seer"]["system_prompt"]
    image_interpreter = llm_factory.create_image_interpreter(
        platform, model_name, system_prompt, CHARACTER["seer"]["temperature"]
    )
    responses, usage = await call_ii(image_interpreter, prompt, None, filepath=filepath)
    image_interpretation = unpack_ii_content(responses[-1]["content"])
    output_string = (
        f"Image Upload, interpreted by {model_name}:\n{image_interpretation}"
    )
    profile["usage"][platform] += usage.total_tokens
    return output_string, profile


async def call_cc(
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
            logger.info(
                "[call_llm]\nResponses: %d\nToken Usage: %s", len(responses), usage
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


async def reply(
    message: telegram.Message, output_string: str
) -> telegram.Message | None:
    from custom_library import escape_html, escape_markdown_extended

    if DEBUG == "1":
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


async def compress_conv_hx(
    llm: Core,
    memory: ShortTermMemory,
) -> tuple[ShortTermMemory, TokenUsage]:
    logger.info("EXEC compress_conv_hx")
    usage = None
    memories = memory.to_list()

    logger.info("len(memories) = %d", len(memories))

    index = -1
    conv_length = 0
    for idx, mem in enumerate(memories, start=0):
        _len = len(mem["content"])
        if _len + conv_length > 32_000:
            index = idx
            break

        conv_length += _len
        logger.info("conv_length = %d", conv_length)

    if index == -1:
        logger.info("<= 32K. LENGTH: %d", conv_length)
        return memory, TokenUsage(input_tokens=0, output_tokens=0)

    # Need Compression
    memory.clear()
    context_strings = []
    for mem in memories[:index]:
        context_strings.append(" - " + mem["role"] + ": " + mem["content"])
    conv_history = "\n".join(context_strings)

    prompt = f"""
    Compress conversations below.

    ---
    {conv_history}
    ---
    """

    try:
        responses, usage = await llm.run_async(query=prompt, context=None)
        # Push the compressed response
        memory.push(
            {
                "role": "assistant",
                "content": f"<compressed>{responses[-1]['content']}</compressed>",
            }
        )
        # Push the rest
        for mem in memories[index:]:
            memory.push(mem)
        logger.info("Compress Token Usage: %s", usage)
        return memory, usage
    except Exception as e:
        logger.error("compress_conv_hx: %s", e)
        # Reduce the oldest
        for mem in memories[1:]:
            memory.push(mem)

        if usage is None:
            usage = TokenUsage(input_tokens=0, output_tokens=0)
        return memory, usage
