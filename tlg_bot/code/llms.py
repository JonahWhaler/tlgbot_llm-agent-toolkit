import logging

from pydantic import BaseModel

from llm_agent_toolkit import Core, ChatCompletionConfig  # type: ignore
from llm_agent_toolkit._core import ImageInterpreter
from llm_agent_toolkit.core import (
    local,
    open_ai,
    deep_seek,
    gemini,
)  #  local aka Ollama
import chromadb

from agent_tools import ToolFactory

from mystorage import WebCache
from myconfig import PARAMETER, PROVIDER, OLLAMA_HOST, CHARACTER

logger = logging.getLogger(__name__)


class LLMFactory:

    def __init__(self, vdb: chromadb.ClientAPI, webcache: WebCache):
        self.tool_factory = ToolFactory(vdb=vdb, web_db=webcache)

    def create_chat_llm(
        self,
        platform: str,
        model_name: str,
        character: str,
        temperature: float,
        structured_output: bool = False,
    ) -> Core:
        supported_providers = ["ollama", "openai", "deepseek", "gemini"]
        if platform not in supported_providers:
            raise ValueError(
                f"Invalid provider. Supported providers: {supported_providers}"
            )

        supported_models = PROVIDER[platform]["t2t"]
        if model_name not in supported_models:
            raise ValueError(f"Invalid model. Supported models: {supported_models}")

        params = PARAMETER["chatcompletion"][platform]
        params["temperature"] = temperature
        logger.info("Creating chat LLM: %s", params)
        config = ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1 if structured_output else 5,
            **params,
        )

        system_prompt = CHARACTER[character]["system_prompt"]

        llm: Core | None
        if platform == "ollama":
            if local.OllamaCore.csv_path is None:
                local.OllamaCore.load_csv("/config/ollama.csv")

            if structured_output:
                llm = local.Text_to_Text_SO(
                    connection_string=OLLAMA_HOST,
                    system_prompt=system_prompt,
                    config=config,
                )
            else:
                tools = CHARACTER[character].get("tools", None)
                tool_list: list | None = None
                if tools:
                    tool_list = []
                    freeuse_llm = local.Text_to_Text(
                        connection_string=OLLAMA_HOST,
                        system_prompt="You are a helpful assistant.",
                        config=config,
                        tools=None,
                    )
                    for t in tools:
                        tool = self.tool_factory.get(tool_name=t, llm=freeuse_llm)
                        if tool is None:
                            raise ValueError(f"Requested tool not found. {t}")
                        tool_list.append(tool)

                llm = local.Text_to_Text(
                    connection_string=OLLAMA_HOST,
                    system_prompt=system_prompt,
                    config=config,
                    tools=tool_list,
                )
        elif platform == "openai":
            if open_ai.OpenAICore.csv_path is None:
                open_ai.OpenAICore.load_csv("/config/openai.csv")
            if config.name.startswith("o1"):
                if config.temperature != 1.0:
                    logger.warning("Force %s temperature to 1.0", config.name)
                    config.temperature = 1.0
                llm = open_ai.O1Beta_OAI_Core(system_prompt="", config=config)
            else:
                if structured_output:
                    llm = open_ai.OAI_StructuredOutput_Core(
                        system_prompt=system_prompt, config=config
                    )
                else:
                    tools = CHARACTER[character].get("tools", None)
                    tool_list: list | None = None
                    if tools:
                        tool_list = []
                        freeuse_llm = open_ai.Text_to_Text(
                            system_prompt="You are a helpful assistant.",
                            config=config,
                            tools=None,
                        )
                        for t in tools:
                            tool = self.tool_factory.get(tool_name=t, llm=freeuse_llm)
                            if tool is None:
                                raise ValueError(f"Requested tool not found. {t}")
                            tool_list.append(tool)

                    llm = open_ai.Text_to_Text(
                        system_prompt=system_prompt, config=config, tools=tool_list
                    )
        elif platform == "gemini":
            if gemini.GeminiCore.csv_path is None:
                gemini.GeminiCore.load_csv("/config/gemini.csv")
            if structured_output:
                llm = gemini.GMN_StructuredOutput_Core(
                    system_prompt=system_prompt, config=config
                )
            else:
                llm = gemini.Text_to_Text(system_prompt=system_prompt, config=config)
        else:  # platform == "deepseek":
            if structured_output:
                llm = deep_seek.Text_to_Text_SO(
                    system_prompt=system_prompt, config=config
                )
            else:
                tools = CHARACTER[character].get("tools", None)
                tool_list: list | None = None
                if tools:
                    tool_list = []
                    freeuse_llm = deep_seek.Text_to_Text(
                        system_prompt="You are a helpful assistant.",
                        config=config,
                        tools=None,
                    )
                    for t in tools:
                        tool = self.tool_factory.get(tool_name=t, llm=freeuse_llm)
                        if tool is None:
                            raise ValueError(f"Requested tool not found. {t}")
                        tool_list.append(tool)
                llm = deep_seek.Text_to_Text(
                    system_prompt=system_prompt, config=config, tools=tool_list
                )
        return llm

    def create_image_interpreter(
        self,
        platform: str,
        model_name: str,
        system_prompt: str,
        temperature: float,
    ) -> ImageInterpreter:
        supported_providers = ["ollama", "openai"]
        if platform not in supported_providers:
            raise ValueError(
                f"Invalid provider. Supported providers: {supported_providers}"
            )

        supported_models = PROVIDER[platform]["i2t"]
        if model_name not in supported_models:
            raise ValueError(f"Invalid model. Supported models: {supported_models}")

        params = PARAMETER["imageinterpretation"][platform]
        params["temperature"] = temperature
        config = ChatCompletionConfig(
            name=model_name, return_n=1, max_iteration=1, **params
        )
        image_interpreter: ImageInterpreter | None
        if platform == "ollama":
            image_interpreter = local.Image_to_Text_SO(
                connection_string=OLLAMA_HOST,
                system_prompt=system_prompt,
                config=config,
            )
        else:  # platform == "openai":
            image_interpreter = open_ai.OAI_StructuredOutput_Core(
                system_prompt=system_prompt, config=config
            )
        return image_interpreter


class Step(BaseModel):
    goal: str
    task: str
    output: str


class LLMResponse(BaseModel):
    prompt: str
    steps: list[Step]
    result: str


class IIResponse(BaseModel):
    long_description: str
    summary: str
    keywords: list[str]
