import os
import re
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
from llm_agent_toolkit.core.local import Text_to_Text
from llm_agent_toolkit import ChatCompletionConfig

from storage import SQLite3_Storage

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
                if sr:
                    r["html"] = sr
        web_search_result = "\n\n".join([json.dumps(r) for r in top_search])
        logger.info("web_search_result: %s", web_search_result)
        return web_search_result

    def run(self, params: str) -> dict:
        # Validate parameters
        if not self.validate(params=params):
            return {"error": "Invalid parameters for DuckDuckGoSearchAgent"}
        # Load parameters
        params = json.loads(params)
        query = params.get("query", None)
        top_n = 5

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
            page = self.fetch(url=r["href"])
            if page:
                r["html"] = page

        web_search_result = "\n\n".join([json.dumps(r) for r in top_search])
        return web_search_result

    async def fetch_async(self, session, url):
        try:
            await asyncio.sleep(self.pause)
            async with session.get(url, headers=self.headers) as response:
                data = await response.text()
                soup = BeautifulSoup(data, "html.parser")
                return self.remove_whitespaces(soup.find("body").text)
        except Exception as _:
            return None

    def fetch(self, url: str):
        try:
            page = requests.get(url=url, headers=self.headers, timeout=2, stream=False)
            soup = BeautifulSoup(page.text, "html.parser")
            body = soup.find("body")
            if body:
                t = body.text
                t = self.remove_whitespaces(t)
                return t
            return None
        except Exception as _:
            return None

    @staticmethod
    def remove_whitespaces(document_content: str) -> str:
        original_len = len(document_content)
        cleaned_text = re.sub(r"\s+", " ", document_content)
        cleaned_text = re.sub(r"\n{3,}", "\n", cleaned_text)
        updated_len = len(cleaned_text)
        logger.info("Reduce from %d to %d", original_len, updated_len)
        return cleaned_text


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
    def __init__(self, vdb: chromadb.ClientAPI, title: str):
        Tool.__init__(self, KnowledgeBaseQueryTool.function_info(title), False)
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


class DDGSmartSearchTool(Tool):
    def __init__(
        self,
        cache_db: SQLite3_Storage,
        safesearch: str = "off",
        region: str = "my-en",
        pause: float = 1.0,
    ):
        Tool.__init__(self, DDGSmartSearchTool.function_info(), True)
        self.safesearch = safesearch
        self.region = region
        self.pause = pause
        self.cache_db = cache_db  # TODO - Handle risk for race condition

        system_prompt = """Task: Summarize web page content, keep what is relevant to the {{Query}}.
        Ensure your answers are grounded.
        """
        self.llm = Text_to_Text(
            connection_string=os.environ["OLLAMA_HOST"],
            system_prompt=system_prompt,
            config=ChatCompletionConfig(
                name="qwen2.5:7b",
                return_n=1,
                max_iteration=1,
                max_tokens=16_000,
                max_output_tokens=2048,
                temperature=0.0,
            ),
            tools=None,
        )
        self.llm.context_length = 32_000
        self.llm.max_output_tokens = 8192

    @staticmethod
    def function_info():
        return FunctionInfo(
            name="DDGSmartSearchTool",
            description="LLM-powered internet search, duckduckgo as backend.",
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
            tasks = []
            for r in top_search:
                tasks.append(self.fetch_async(session, r["href"]))

            search_results = await asyncio.gather(*tasks)
            for r, sr in zip(top_search, search_results):
                if sr:
                    if len(sr) < 1024:
                        r["html"] = sr
                    else:
                        prompt = f"""Query: {query}
                        Here is the web search result:
                        ---
                        * Page Title: {r['title']}
                        * Page Body: {r['body']}
                        * Page Content: {sr[:8_000]}
                        ---
                        """
                        responses = await self.llm.run_async(query=prompt, context=None)
                        response = responses[-1]
                        summarized_content = response["content"]
                        logger.info("Sumarized page content: %s", summarized_content)
                        r["html"] = summarized_content
                        
        web_search_result = "\n\n".join([json.dumps(r) for r in top_search])
        return web_search_result

    def run(self, params: str) -> dict:
        # Validate parameters
        if not self.validate(params=params):
            return {"error": "Invalid parameters for DuckDuckGoSearchAgent"}
        # Load parameters
        params = json.loads(params)
        query = params.get("query", None)
        top_n = 5

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
            page = self.fetch(url=r["href"])
            if page:
                page_content_len = len(page)
                if page_content_len < 1024:
                    r["html"] = page
                else:
                    prompt = f"""Query: {query}
                    Here is the web search result:
                    ---
                    * Page Title: {r['title']}
                    * Page Body: {r['body']}
                    * Page Content: {page[:8_000]}
                    ---
                    """
                    response = self.llm.run(query=prompt, context=None)[0]
                    summarized_content = response["content"]
                    logger.info("Sumarized page content: %s", summarized_content)
                    r["html"] = summarized_content

        web_search_result = "\n\n".join([json.dumps(r) for r in top_search])
        return web_search_result

    async def fetch_async(self, session, url):
        try:
            # Load from cache
            result = self.cache_db.get(url)
            if result:
                return result
            
            await asyncio.sleep(self.pause)
            async with session.get(url, headers=self.headers) as response:
                data = await response.text()
                soup = BeautifulSoup(data, "html.parser")
                output_string = self.remove_whitespaces(soup.find("body").text)
                # Store to cache
                self.cache_db.set(url, output_string)
                return output_string
        except Exception as _:
            return None

    def fetch(self, url: str):
        try:
            # Load from cache
            result = self.cache_db.get(url)
            if result:
                return result

            page = requests.get(url=url, headers=self.headers, timeout=2, stream=False)
            soup = BeautifulSoup(page.text, "html.parser")
            body = soup.find("body")
            if body:
                t = body.text
                t = self.remove_whitespaces(t)
                # Store to cache
                self.cache_db.set(url, t)
                return t
            return None
        except Exception as _:
            return None

    @staticmethod
    def remove_whitespaces(document_content: str) -> str:
        original_len = len(document_content)
        cleaned_text = re.sub(r"\s+", " ", document_content)
        cleaned_text = re.sub(r"\n{3,}", "\n", cleaned_text)
        updated_len = len(cleaned_text)
        logger.info("Reduce from %d to %d", original_len, updated_len)
        return cleaned_text


class ToolFactory:
    def __init__(self, vdb: chromadb.ClientAPI, web_db: SQLite3_Storage):
        self.vdb = vdb
        self.web_db = web_db  # <-- Risk for Race Condition!!!

    def get(self, tool_name: str) -> Tool | None:
        if tool_name == "current_datetime":
            return LazyTool(function=current_datetime, is_coroutine_function=False)
        if tool_name == "duckduckgo_search":
            return DuckDuckGoSearchTool()
        if tool_name == "knowledge_base_query:tzuchi":
            logger.info("Initializing KnowledgeBaseQueryTool...")
            _tool = KnowledgeBaseQueryTool(vdb=self.vdb, title="tzuchi")
            logger.info("KnowledgeBaseQueryTool initialized.")
            return _tool
        if tool_name == "ddgsmart_search":
            return DDGSmartSearchTool(cache_db=self.web_db)
        return None
