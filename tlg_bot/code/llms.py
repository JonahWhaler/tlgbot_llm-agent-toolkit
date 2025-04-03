import logging
from typing import Optional
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

# from agent_tools import ToolFactory
from toolbox import ToolFactory

from mystorage import WebCache
from myconfig import PARAMETER, PROVIDER, OLLAMA_HOST, CHARACTER

logger = logging.getLogger(__name__)


class LLMFactory:

    def __init__(
        self,
        vdb: chromadb.ClientAPI,
        webcache: WebCache,
        user_vdb: chromadb.ClientAPI | None = None,
        encoder_config: dict | None = None,
    ):
        self.tool_factory = ToolFactory(
            vdb=vdb, web_db=webcache, user_vdb=user_vdb, encoder_config=encoder_config
        )

    def __create_openai_chat_llm(
        self,
        model_name: str,
        system_prompt: str,
        config: ChatCompletionConfig,
        tools: Optional[list] = None,
        structured_output: bool = False,
        **kwargs,
    ) -> Core:
        if open_ai.OpenAICore.csv_path is None:
            open_ai.OpenAICore.load_csv("/config/openai.csv")

        if model_name in ["o1-mini", "o3-mini"]:
            reasoning_effort = kwargs.get("reasoning_effort", "medium")
            config.temperature = 1.0
            return open_ai.Reasoning_Core(
                system_prompt=system_prompt,
                config=config,
                reasoning_effort=reasoning_effort,
            )
        if structured_output:
            return open_ai.StructuredOutput(system_prompt=system_prompt, config=config)

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
                    logger.warning("Requested tool not found. %s", t)
                    continue
                tool_list.append(tool)

        return open_ai.Text_to_Text(
            system_prompt=system_prompt, config=config, tools=tool_list
        )

    def __create_deepseek_chat_llm(
        self,
        model_name: str,
        system_prompt: str,
        config: ChatCompletionConfig,
        tools: Optional[list] = None,
        structured_output: bool = False,
        **kwargs,
    ) -> Core:
        if model_name in ["deepseek-reasoner"]:
            return deep_seek.Reasoner_Core(system_prompt=system_prompt, config=config)

        if structured_output:
            return deep_seek.Text_to_Text_SO(system_prompt=system_prompt, config=config)

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
                    logger.warning("Requested tool not found. %s", t)
                    continue
                tool_list.append(tool)

        return deep_seek.Text_to_Text(system_prompt=system_prompt, config=config)

    def __create_gemini_chat_llm(
        self,
        model_name: str,
        system_prompt: str,
        config: ChatCompletionConfig,
        tools: Optional[list] = None,
        structured_output: bool = False,
        **kwargs,
    ) -> Core:
        if gemini.GeminiCore.csv_path is None:
            gemini.GeminiCore.load_csv("/config/gemini.csv")

        if model_name in [
            "gemini-2.0-flash-thinking-exp-01-21",
            "gemini-2.5-pro-exp-03-25",
        ]:
            return gemini.Thinking_Core(system_prompt=system_prompt, config=config)

        if structured_output:
            return gemini.StructuredOutput(system_prompt=system_prompt, config=config)

        tool_list: list | None = None
        if tools:
            tool_list = []
            freeuse_llm = gemini.Text_to_Text(
                system_prompt="You are a helpful assistant.", config=config
            )
            for t in tools:
                tool = self.tool_factory.get(tool_name=t, llm=freeuse_llm)
                if tool is None:
                    logger.warning("Requested tool not found. %s", t)
                    continue
                tool_list.append(tool)

        return gemini.Text_to_Text_W_Tool(
            system_prompt=system_prompt, config=config, tools=tool_list
        )

    def __create_ollama_chat_llm(
        self,
        model_name: str,
        system_prompt: str,
        config: ChatCompletionConfig,
        tools: Optional[list] = None,
        structured_output: bool = False,
        **kwargs,
    ) -> Core:
        if local.OllamaCore.csv_path is None:
            local.OllamaCore.load_csv("/config/ollama.csv")

        if structured_output:
            return local.Image_to_Text_SO(
                connection_string=OLLAMA_HOST,
                system_prompt=system_prompt,
                config=config,
            )

        tool_list: list | None = None
        if tools:
            tool_list = []
            freeuse_llm = local.Text_to_Text(
                connection_string=OLLAMA_HOST,
                system_prompt="You are a helpful assistant.",
                config=config,
            )
            for t in tools:
                tool = self.tool_factory.get(tool_name=t, llm=freeuse_llm)
                if tool is None:
                    logger.warning("Requested tool not found. %s", t)
                    continue
                tool_list.append(tool)

        return local.Text_to_Text(
            connection_string=OLLAMA_HOST,
            system_prompt=system_prompt,
            config=config,
            tools=tool_list,
        )

    def create_chat_llm(
        self,
        platform: str,
        model_name: str,
        character: str,
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
        params["temperature"] = CHARACTER[character]["temperature"]
        logger.info("Creating chat LLM: %s", params)
        config = ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1 if structured_output else 3,
            **params,
        )

        system_prompt = CHARACTER[character]["system_prompt"]
        tools = CHARACTER[character].get("tools", None)
        llm: Core | None
        if platform == "ollama":
            llm = self.__create_ollama_chat_llm(
                model_name=model_name,
                system_prompt=system_prompt,
                config=config,
                tools=tools,
                structured_output=structured_output,
            )
        elif platform == "openai":
            llm = self.__create_openai_chat_llm(
                model_name=model_name,
                system_prompt=system_prompt,
                config=config,
                tools=tools,
                structured_output=structured_output,
            )
        elif platform == "gemini":
            llm = self.__create_gemini_chat_llm(
                model_name=model_name,
                system_prompt=system_prompt,
                config=config,
                tools=tools,
                structured_output=structured_output,
            )
        else:  # platform == "deepseek":
            llm = self.__create_deepseek_chat_llm(
                model_name=model_name,
                system_prompt=system_prompt,
                config=config,
                tools=tools,
                structured_output=structured_output,
            )
        return llm

    def create_image_interpreter(
        self,
        platform: str,
        model_name: str,
        system_prompt: str,
        temperature: float,
    ) -> ImageInterpreter:
        supported_providers = ["ollama", "openai", "gemini"]
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
            if local.OllamaCore.csv_path is None:
                local.OllamaCore.load_csv("/config/ollama.csv")

            image_interpreter = local.Image_to_Text_SO(
                connection_string=OLLAMA_HOST,
                system_prompt=system_prompt,
                config=config,
            )
        elif platform == "gemini":
            if gemini.GeminiCore.csv_path is None:
                gemini.GeminiCore.load_csv("/config/gemini.csv")

            image_interpreter = gemini.GMN_StructuredOutput_Core(
                system_prompt=system_prompt, config=config
            )
        else:  # platform == "openai":
            if open_ai.OpenAICore.csv_path is None:
                open_ai.OpenAICore.load_csv("/config/openai.csv")

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
