import os
import json
import logging
from typing import Optional
from datetime import datetime
from telegram import Message
from telegram.ext import CallbackContext
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
            "system_prompt": CHARACTER[character]["system_prompt"],
        }
        t = CHARACTER[character].get("tools", None)
        # heavily rely on the tool name, no further description here.
        if t:
            agent_object["tools"] = t
        agents.append(agent_object)

    input_prompt = {"request": prompt, "agents": agents}
    responses, usage = await agent_router.run_async(
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
        return best_agent, usage
    except json.JSONDecodeError:
        return DEFAULT_CHARACTER


async def update_memory(
    agent: Core,
    cmemory: dict[str, ShortTermMemory],
    db: mystorage.SQLite3_Storage,
    identifier: str,
    new_content: str | None,
) -> TokenUsage | None:
    umemory: Optional[ShortTermMemory] = cmemory.get(identifier, None)
    if new_content and umemory is not None:
        if not new_content.startswith("/"):
            umemory.push({"role": "user", "content": new_content})
            cmemory[identifier] = umemory

    uprofile = db.get(identifier)
    if DEBUG == "1":
        assert uprofile is not None

    old_interactions = umemory.to_list()[:MEMORY_LEN]
    if len(old_interactions) == 0:
        return

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
    existing_memory = uprofile["memory"]
    instructions = []
    if existing_memory:
        instructions.extend(
            [
                "Instruction:\nUpdate user's metadata.",
                f"Existing Metadata: {existing_memory}",
            ]
        )
    else:
        instructions.append("Construct user's metadata.")

    if len(selected_interactions) > 0:
        instructions.append(f"Recent Interactions: {json.dumps(selected_interactions)}")

    prompt = "\n\n".join(instructions)
    responses, usage = await agent.run_async(
        query=prompt, context=None, mode=ResponseMode.JSON
    )
    uprofile["memory"] = responses[0]["content"]
    db.set(identifier, uprofile)
    return usage


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
    await store_to_drive(file_id, temp_path, context, False)

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
    MAX_RETRY = 5
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
            if context and str(ve) == "max_output_tokens <= 0":
                context = context[1:]
                iteration += 1
            else:
                raise
    raise ValueError(f"max_output_tokens <= 0. Retry up to {MAX_RETRY} times.")


async def image_interpreter_pipeline(
    profile: dict, prompt: str, filepath: str, llm_factory: LLMFactory
) -> tuple[str, TokenUsage]:
    logger.info("Execute image_interpreter_pipeline")
    platform = profile["platform_i2t"]
    model_name = profile["model_i2t"]
    system_prompt = CHARACTER["seer"]["system_prompt"]
    image_interpreter = llm_factory.create_image_interpreter(
        platform, model_name, system_prompt, profile["creativity"]
    )
    responses, usage = await call_ii(image_interpreter, prompt, None, filepath=filepath)
    image_interpretation = unpack_ii_content(responses[-1]["content"])
    output_string = (
        f"Image Upload, interpreted by {model_name}:\n{image_interpretation}"
    )
    return output_string, usage
