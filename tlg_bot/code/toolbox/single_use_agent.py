import logging
import json

from llm_agent_toolkit import (
    Core,
    Tool,
    FunctionInfo,
    FunctionProperty,
    FunctionPropertyType,
    FunctionParameters,
    ChatCompletionConfig,
)
from llm_agent_toolkit._tool import FunctionParameterConstraint
from llm_agent_toolkit.core import gemini, deep_seek, open_ai


class SingleUseAgent(Tool):
    """
    # Single Use Agent

    Notes:
    - Current implementation only supports Gemini, DeepSeek and OpenAI
    - The created agent has no function calling capability
    - The created agent has no memory/context

    Warning:
    - **Token Usage**: Token usage is tracked but grouped by the provider.
    """

    def __init__(self):
        Tool.__init__(self, SingleUseAgent.function_info(), True)

    @staticmethod
    def function_info():
        description = """
        Define a single use llm agent and delegate work to the agent.
        This allows the main LLM focus on the execution orders by delegating the sub-problems
        to the agent.

        Notes:
        - Current implementation only supports Gemini, DeepSeek and OpenAI
        - The created agent has no function calling capability
        - The created agent has no memory/context
        """
        return FunctionInfo(
            name="SingleUseAgent",
            description=description,
            parameters=FunctionParameters(
                properties=[
                    FunctionProperty(
                        name="system_prompt",
                        type=FunctionPropertyType.STRING,
                        description="System instruction of the agent.",
                    ),
                    FunctionProperty(
                        name="question_or_task",
                        type=FunctionPropertyType.STRING,
                        description="Questions to answer or a task to perform.",
                    ),
                    FunctionProperty(
                        name="model",
                        type=FunctionPropertyType.STRING,
                        description="LLM model",
                        constraint=FunctionParameterConstraint(
                            enum=["gemini-1.5-flash", "gpt-4o-mini", "deepseek-chat"]
                        ),
                    ),
                    FunctionProperty(
                        name="temperature",
                        type=FunctionPropertyType.NUMBER,
                        description="Temperature of the LLM.",
                        constraint=FunctionParameterConstraint(
                            minimum=0.0, maximum=1.0
                        ),
                    ),
                ],
                type="object",
                required=[
                    "system_prompt",
                    "question_or_task",
                ],
            ),
        )

    def create_llm(
        self, model: str, system_prompt: str, temperature: float = 0.7
    ) -> Core:
        config = ChatCompletionConfig(
            name=model,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=2048,
            temperature=temperature,
        )
        if "gemini" in model:
            return gemini.Text_to_Text(system_prompt=system_prompt, config=config)

        if "gpt" in model:
            return open_ai.Text_to_Text(system_prompt=system_prompt, config=config)

        # Default
        # if "deepseek" in model:
        return deep_seek.Text_to_Text(system_prompt=system_prompt, config=config)

    async def run_async(self, params: str) -> str:
        logger = logging.getLogger(__name__)

        params_dict = json.loads(params)
        valid, validation_message = self.validate(**params_dict)
        if not valid:
            return json.dumps(
                {
                    "error": "Invalid Parameters",
                    "detail": validation_message,
                },
                ensure_ascii=False,
            )
        # Load parameters
        system_prompt: str = params_dict.get("system_prompt", None)
        question_or_task: str = params_dict.get("question_or_task", None)
        model: str = params_dict.get("model", "deepseek-chat")
        temperature: float = params_dict.get("temperature", 0.7)

        llm = self.create_llm(
            model=model, system_prompt=system_prompt, temperature=temperature
        )
        try:
            responses, usage = await llm.run_async(query=question_or_task, context=None)
            self.token_usage = self.token_usage + usage
            return responses[-1]["content"]
        except Exception as e:
            logger.error("SingleUseAgent: %s", str(e), exc_info=True, stack_info=True)
            return json.dumps(
                {"error": "Failed", "detail": str(e)},
                ensure_ascii=False,
            )

    def run(self, params: str) -> str:
        logger = logging.getLogger(__name__)
        params_dict = json.loads(params)
        valid, validation_message = self.validate(**params_dict)
        if not valid:
            return json.dumps(
                {
                    "error": "Invalid Parameters",
                    "detail": validation_message,
                },
                ensure_ascii=False,
            )
        # Load parameters
        system_prompt: str = params_dict.get("system_prompt", None)
        question_or_task: str = params_dict.get("question_or_task", None)
        model: str = params_dict.get("model", "deepseek-chat")
        temperature: float = params_dict.get("temperature", 0.7)

        llm = self.create_llm(
            model=model, system_prompt=system_prompt, temperature=temperature
        )
        try:
            responses, usage = llm.run(query=question_or_task, context=None)
            self.token_usage = self.token_usage + usage
            return responses[-1]["content"]
        except Exception as e:
            logger.error("SingleUseAgent: %s", str(e), exc_info=True, stack_info=True)
            return json.dumps(
                {"error": "Failed", "detail": str(e)},
                ensure_ascii=False,
            )
