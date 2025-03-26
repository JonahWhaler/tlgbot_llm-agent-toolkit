from datetime import datetime

import chromadb
from llm_agent_toolkit import Core, Tool
from llm_agent_toolkit.tool import LazyTool

# DuckDuckGo Search
from .ddg_search import DuckDuckGoSearchTool
from .ddg_smart_search import DDGSmartSearchTool

# Google Search
from .g_search import GoogleSearchTool
from .g_smart_search import GoogleSmartSearchTool

# Knowledge
from .topic_knowledge import TopicQueryTool
from .personal_knowledge import PersonalKnowledgeBaseTool

from .single_use_agent import SingleUseAgent

from mystorage import WebCache


def current_datetime(timezone: str = "Asia/Kuala_Lumpur"):
    """
    Retrieve current datetime with respected timezone.

    Args:
        timezone (str): The timezone to use, default Asia/Kuala_Lumpur
                        Should comply with the code defined here: https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568

    Returns:
        str: Current datetime with respected timezone
    """
    import pytz

    return datetime.now(tz=pytz.timezone(timezone)).strftime("%Y-%m-%d %H:%M:%S")


class ToolFactory:
    def __init__(
        self,
        vdb: chromadb.ClientAPI,
        web_db: WebCache,
        user_vdb: chromadb.ClientAPI | None = None,
        encoder_config: dict | None = None,
    ):
        self.vdb = vdb
        self.web_db = web_db
        self.user_vdb = user_vdb
        self.encoder_config = encoder_config

    def get(self, tool_name: str, llm: Core | None = None) -> Tool | None:
        if tool_name == "current_datetime":
            return LazyTool(function=current_datetime, is_coroutine_function=False)

        if tool_name == "duckduckgo_search":
            # Set a lower num_results to reduce load to the global context
            return DuckDuckGoSearchTool(web_cache=self.web_db, num_results=3)

        if tool_name == "ddgsmart_search":
            if llm is None:
                raise ValueError("LLM is required for DDGSmartSearchTool")
            return DDGSmartSearchTool(llm=llm, web_cache=self.web_db)

        if tool_name == "google_search":
            return GoogleSearchTool(web_cache=self.web_db, num_results=5)

        if tool_name == "google_smart_search":
            if llm is None:
                raise ValueError("LLM is required for GoogleSmartSearchTool")
            return GoogleSmartSearchTool(llm=llm, web_cache=self.web_db, num_results=5)

        if tool_name == "topic_query:tzuchi":
            assert (
                self.encoder_config is not None
            ), "encoder_config is required for TopicQueryTool"
            _tool = TopicQueryTool(
                vdb=self.vdb,
                title="tzuchi",
                encoder_config=self.encoder_config,
                num_results=10,
            )
            return _tool

        if tool_name == "personal_knowledge_base":
            assert (
                self.user_vdb is not None
            ), "user_vdb is required for PersonalKnowledgeBaseTool"
            return PersonalKnowledgeBaseTool(
                user_vdb=self.user_vdb,
                encoder_config=self.encoder_config,
                num_results=10,
            )

        if tool_name == "single_use_agent":
            return SingleUseAgent()

        return None
