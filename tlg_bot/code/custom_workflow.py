import json
import logging
from typing import Optional

from llm_agent_toolkit import Core, ResponseMode, ShortTermMemory

import storage
from config import CHARACTER, DEFAULT_CHARACTER, MEMORY_LEN, DEBUG

logger = logging.getLogger(__name__)


async def find_best_agent(
    agent_router: Core, prompt: str, context: list[dict] | None
) -> str:
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


async def update_memory(
    agent: Core,
    cmemory: ShortTermMemory,
    db: storage.SQLite3_Storage,
    identifier: str,
    new_content: str | None,
) -> str | None:
    umemory: Optional[ShortTermMemory] = cmemory.get(identifier, None)
    if new_content and umemory is not None:
        if not new_content.startswith("/"):
            umemory.push({"role": "user", "content": new_content})

    uprofile = db.get(identifier)
    if DEBUG == "1":
        assert uprofile is not None

    old_interactions = umemory.to_list()[:MEMORY_LEN]
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

    responses = await agent.run_async(
        query=prompt, context=context, mode=ResponseMode.JSON
    )
    uprofile["memory"] = responses[0]["content"]
    db.set(identifier, uprofile)
