"""
Routing Strategies
"""

import json
import logging
from typing import Optional, Protocol, runtime_checkable
from pydantic import BaseModel
from llm_agent_toolkit import ChatCompletionConfig, Core, ResponseMode
from llm_agent_toolkit.core import open_ai, gemini
from llm_agent_toolkit._util import TokenUsage

from myconfig import CHARACTER

logger = logging.getLogger(__name__)


class AgentProfile(BaseModel):
    name: str
    primary_role: str
    description: str
    suitable_tasks: list[str]
    unsuitable_tasks: list[str]
    tools: Optional[list[str]] = None


class SelectedAgent(BaseModel):
    name: str
    reason: str
    relevant_score: float


class RouterResponse(BaseModel):
    agents: list[SelectedAgent]


@runtime_checkable
class RouteStrategy(Protocol):
    async def route(
        self,
        prompt: str,
        context: list[dict] | None,
        agents_information: list[AgentProfile],
    ) -> tuple[RouterResponse, TokenUsage]: ...


def create_router(
    provider: str, model_name: str, config: dict, system_prompt: str
) -> Core:
    if provider not in ["openai", "gemini"]:
        raise ValueError("Invalid provider. Supported providers: ['openai', 'gemini']")

    config = ChatCompletionConfig(name=model_name, **config)
    if provider == "openai":
        return open_ai.StructuredOutput(system_prompt=system_prompt, config=config)
    return gemini.StructuredOutput(system_prompt=system_prompt, config=config)


class OneShotRouting:
    def __init__(self, provider: str, model_name: str, top_n: int = 1):
        system_prompt = CHARACTER["router_oneshot"]["system_prompt"]
        config = {
            "temperature": CHARACTER["router_oneshot"]["temperature"],
            "return_n": 1,
            "max_iteration": 1,
            "max_token": 2048,
            "max_output_tokens": 1024,
        }
        self.router = create_router(provider, model_name, config, system_prompt)
        self.top_n = top_n

    async def route(
        self,
        prompt: str,
        context: list[dict] | None,
        agents_information: list[AgentProfile],
    ) -> tuple[RouterResponse, TokenUsage]:
        # Step 1 - Build structured input
        structured_input = {
            "request": prompt,
            "agents": [
                agent.model_dump_json(exclude_none=True) for agent in agents_information
            ],
            "task": f"Pick top-{self.top_n} agent/s relevant to user's request.",
        }
        if context:
            structured_input["context"] = context

        # Step 2 - Run
        response_tuple: tuple[list[dict], TokenUsage] = await self.router.run_async(
            query=json.dumps(structured_input, ensure_ascii=False),
            context=None,
            mode=ResponseMode.SO,
            format=RouterResponse,
        )
        responses, token_usage = response_tuple

        # Step 3 - Validate response structure
        response = responses[-1]
        json_response = json.loads(response["content"])
        agents = json_response["agents"]
        if not isinstance(agents, list):
            raise RuntimeError(
                "Malformed Response: Router did not return a list of agents."
            )

        if len(agents) != self.top_n:
            raise RuntimeError(
                f"Malformed Response: Expect top-{self.top_n} agents, get {len(agents)}."
            )

        for agent in agents:
            for k in ["name", "reason", "relevant_score"]:
                if k not in agent:
                    raise RuntimeError(f"Malformed Response: Missing {k}.")
                if not isinstance(agent["relevant_score"], float):
                    raise RuntimeError(
                        "Malformed Response: relevant_score is not a float."
                    )

        # Step 4 - Build RouterResponse
        agent_list = []
        for agent in agents:
            agent_list.append(
                SelectedAgent(
                    name=agent["name"],
                    reason=agent["reason"],
                    relevant_score=agent["relevant_score"],
                )
            )
        router_response = RouterResponse(agents=agent_list)
        return router_response, token_usage


class OneByOneRouting:
    def __init__(
        self,
        provider: str,
        model_name: str,
        top_n: int = 1,
    ):
        system_prompt = CHARACTER["router_onebyone"]["system_prompt"]
        config = {
            "temperature": CHARACTER["router_onebyone"]["temperature"],
            "return_n": 1,
            "max_iteration": 1,
            "max_token": 2048,
            "max_output_tokens": 1024,
        }
        self.router = create_router(provider, model_name, config, system_prompt)
        self.top_n = top_n

    async def route(
        self,
        prompt: str,
        context: list[dict] | None,
        agents_information: list[AgentProfile],
    ) -> tuple[RouterResponse, TokenUsage]:
        agent_list = []
        for agent in agents_information:
            # Step 1 - Build structured input
            structured_input = {
                "request": prompt,
                "agent": agent.model_dump_json(exclude_none=True),
                "task": "Grade the agent's relevance to user's request.",
            }
            if context:
                structured_input["context"] = context

            # Step 2 - Run
            response_tuple: tuple[list[dict], TokenUsage] = await self.router.run_async(
                query=json.dumps(structured_input, ensure_ascii=False),
                context=None,
                mode=ResponseMode.SO,
                format=SelectedAgent,
            )
            responses, token_usage = response_tuple

            # Step 3 - Validate response structure
            response = responses[-1]
            json_response = json.loads(response["content"])
            # agent = json_response
            if not isinstance(json_response, dict):
                raise RuntimeError(
                    "Malformed Response: Router did not return a dict of agent."
                )

            for k in ["name", "reason", "relevant_score"]:
                if k not in json_response:
                    raise RuntimeError(f"Malformed Response: Missing {k}.")

                if not isinstance(json_response["relevant_score"], float):
                    raise RuntimeError(
                        "Malformed Response: relevant_score is not a float."
                    )

            # Step 4 - Build SelectedAgent
            agent_list.append(
                SelectedAgent(
                    name=agent.name,
                    reason=json_response["reason"],
                    relevant_score=json_response["relevant_score"],
                )
            )

        agent_list.sort(key=lambda obj: obj.relevant_score, reverse=True)

        # Step 5 - Build RouterResponse
        router_response = RouterResponse(agents=agent_list[: self.top_n])
        return router_response, token_usage
