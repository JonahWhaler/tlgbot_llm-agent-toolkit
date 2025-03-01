import os
import logging
import random
import json
import asyncio
from datetime import datetime
import aiohttp
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import llm_agent_toolkit
import requests
import chromadb
from llm_agent_toolkit import (
    Tool,
    FunctionInfo,
    FunctionProperty,
    FunctionPropertyType,
    FunctionParameters,
)
from llm_agent_toolkit.tool import LazyTool
from llm_agent_toolkit.memory import ChromaMemory
from util import ChromaDBFactory
import config

logger = logging.getLogger(__name__)


class DuckDuckGoSearchTool(Tool):
    def __init__(
        self, safesearch: str = "off", region: str = "my-en", pause: float = 1.0
    ):
        Tool.__init__(self, DuckDuckGoSearchTool.function_info(), True)
        self.safesearch = safesearch
        self.region = region
        self.pause = pause

    @staticmethod
    def function_info():
        return FunctionInfo(
            name="DuckDuckGoSearchTool",
            description="Search the internet via DuckDuckGO API.",
            parameters=FunctionParameters(
                properties=[
                    FunctionProperty(
                        name="query",
                        type=FunctionPropertyType.STRING,
                        description="Keyword that describe or define the query",
                    )
                ],
                type="object",
                required=["query"],
            ),
        )

    @property
    def random_user_agent(self) -> str:
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) Firefox/120.0",
        ]
        return random.choice(user_agents)

    @property
    def headers(self) -> dict:
        return {
            "User-Agent": self.random_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

    async def run_async(self, params: str) -> dict:
        # Validate parameters
        if not self.validate(params=params):
            return {"error": "Invalid parameters for DuckDuckGoSearchAgent"}
        # Load parameters
        params = json.loads(params)
        query = params.get("query", None)
        top_n = 5

        output = {}
        top_search = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                keywords=query,
                region=self.region,
                safesearch=self.safesearch,
                max_results=top_n,
            ):
                top_search.append(r)
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_async(session, r["href"]) for r in top_search]
            search_results = await asyncio.gather(*tasks)
            for r, sr in zip(top_search, search_results):
                r["html"] = sr
        web_search_result = "\n\n".join([json.dumps(r) for r in top_search])
        output["result"] = [
            ("web_search_result", web_search_result, datetime.now().isoformat(), True)
        ]
        return str(output)

    def run(self, params: str) -> dict:
        # Validate parameters
        if not self.validate(params=params):
            return {"error": "Invalid parameters for DuckDuckGoSearchAgent"}
        # Load parameters
        params = json.loads(params)
        query = params.get("query", None)
        top_n = 5

        output = {}
        top_search = []
        with DDGS() as ddgs:
            try:
                for r in ddgs.text(
                    keywords=query,
                    region=self.region,
                    safesearch=self.safesearch,
                    max_results=top_n,
                ):
                    top_search.append(r)
            except Exception as error:
                logger.error(error)

        for r in top_search:
            r["html"] = self.fetch(url=r["href"])

        web_search_result = "\n\n".join([json.dumps(r) for r in top_search])
        output["result"] = [
            ("web_search_result", web_search_result, datetime.now().isoformat(), True)
        ]
        return str(output)

    async def fetch_async(self, session, url):
        try:
            await asyncio.sleep(self.pause)
            async with session.get(url, headers=self.headers) as response:
                data = await response.text()
                soup = BeautifulSoup(data, "html.parser")
                return soup.find("body").text
        except Exception as _:
            return "Webpage not available, either due to an error or due to lack of access permissions to the site."

    def fetch(self, url: str):
        try:
            page = requests.get(url=url, headers=self.headers, timeout=2, stream=False)
            soup = BeautifulSoup(page.text, "html.parser")
            body = soup.find("body")
            if body:
                t = body.text
                t = t.replace("\n\n", "\n")
                t = t.replace("\\n\n", "\\n")
                return t
            return "Webpage not available, either due to an error or due to lack of access permissions to the site."
        except Exception as _:
            return "Webpage not available, either due to an error or due to lack of access permissions to the site."


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


class KnowledgeBaseQueryTool(Tool):
    def __init__(self, title: str, directory: str):
        Tool.__init__(self, KnowledgeBaseQueryTool.function_info(title), False)
        vdb: chromadb.ClientAPI = ChromaDBFactory.get_instance(
            persist=True, persist_directory=directory
        )
        try:
            c = vdb.get_collection(name=title)
            logger.info("Count: %d", c.count())
            if c.count() == 0:
                logger.info("Creating new collection: %s", title)
                self.init(vdb, title)
                logger.info("Collection created: %s", title)
        except Exception as e:
            logger.error("Error creating knowledge base: %s", e)
        finally:
            encoder = llm_agent_toolkit.encoder.OllamaEncoder(
                connection_string=config.OLLAMA_HOST, model_name="bge-m3:latest"
            )
            # choice of chunker is not important
            # query does not need to be chunked
            chunker = llm_agent_toolkit.chunkers.FixedGroupChunker(
                config={"K": 1, "resolution": "back", "level": "character"}
            )
            chroma_memory = ChromaMemory(
                vdb=vdb, encoder=encoder, chunker=chunker, namespace=title
            )
            self.knowledge_base = chroma_memory

    def init(self, vdb, title):
        encoder = llm_agent_toolkit.encoder.OllamaEncoder(
            connection_string=config.OLLAMA_HOST, model_name="bge-m3:latest"
        )
        for file_name in os.listdir(f"/assets/{title}"):
            file_path = f"/assets/{title}/{file_name}"
            logger.info("Loading file: %s", file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                K = max(len(text) // (encoder.ctx_length // 2), 1)
                if K == 1:
                    chunker = llm_agent_toolkit.chunkers.FixedGroupChunker(
                        config={"K": 1}
                    )
                else:
                    chunker = llm_agent_toolkit.chunkers.SemanticChunker(
                        encoder=encoder,
                        config={
                            "K": K,
                            "MAX_ITERATION": 50,
                            "update_rate": 0.3,
                            "min_coverage": 0.95,
                        },
                    )

                ChromaMemory(
                    vdb=vdb,
                    encoder=encoder,
                    chunker=chunker,
                    namespace=title,
                    overwrite=False,
                ).add(document_string=text)

    @staticmethod
    def function_info(title: str):
        return FunctionInfo(
            name="KnowledgeBaseQueryTool",
            description=f"Query from {title} knowledge base.",
            parameters=FunctionParameters(
                properties=[
                    FunctionProperty(
                        name="query",
                        type=FunctionPropertyType.STRING,
                        description="Keyword that describe or define the query",
                    )
                ],
                type="object",
                required=["query"],
            ),
        )

    def run(self, params: str) -> str:
        params = json.loads(params)
        query = params.get("query", None)
        if query:
            return self.knowledge_base.query(
                query, return_n=5, output_types=["documents"]
            )["result"]["documents"]
        return "No query provided"

    async def run_async(self, params: str) -> str:
        return self.run(params)


class ToolFactory:
    def __init__(self):
        pass

    def get(self, tool_name: str) -> Tool | None:
        if tool_name == "current_datetime":
            return LazyTool(function=current_datetime, is_coroutine_function=False)
        if tool_name == "duckduckgo_search":
            return DuckDuckGoSearchTool()
        if tool_name == "knowledge_base_query:tzuchi":
            logger.info("Initializing KnowledgeBaseQueryTool...")
            _tool = KnowledgeBaseQueryTool(title="tzuchi", directory="/temp/vect")
            logger.info("KnowledgeBaseQueryTool initialized.")
            return _tool
        return None
