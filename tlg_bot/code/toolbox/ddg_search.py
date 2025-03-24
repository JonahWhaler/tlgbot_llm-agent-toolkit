import re
import logging
import random
import json
import asyncio
import time
import aiohttp
import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from llm_agent_toolkit import (
    Tool,
    FunctionInfo,
    FunctionProperty,
    FunctionPropertyType,
    FunctionParameters,
)


class DuckDuckGoSearchTool(Tool):
    def __init__(
        self,
        safesearch: str = "off",
        region: str = "my-en",
        pause: float = 1.0,
        num_results: int = 5,
    ):
        Tool.__init__(self, DuckDuckGoSearchTool.function_info(), True)
        self.safesearch = safesearch
        self.region = region
        self.pause = pause
        self.num_results = num_results

    @staticmethod
    def function_info():
        description = """
        Search for the relevant pages via DuckDuckGo API.
        Naive implementation:
        - results are not cached.
        - Text only.
        """
        return FunctionInfo(
            name="DuckDuckGoSearchTool",
            description=description,
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

    @staticmethod
    def remove_whitespaces(document_content: str) -> str:
        cleaned_text = re.sub(r"\s+", " ", document_content)
        cleaned_text = re.sub(r"\n{3,}", "\n", cleaned_text)
        return cleaned_text

    async def run_async(self, params: str) -> dict:
        logger = logging.getLogger(__name__)
        # Validate parameters
        if not self.validate(params=params):
            return json.dumps(
                {"error": "Invalid parameters for DuckDuckGoSearchAgent"},
                ensure_ascii=True,
            )
        # Load parameters
        params = json.loads(params)
        query = params.get("query", None)

        top_search = []
        error_message = "Unknown Error"
        with DDGS(headers=self.headers) as ddgs:
            try:
                for r in ddgs.text(
                    keywords=query,
                    region=self.region,
                    safesearch=self.safesearch,
                    max_results=self.num_results,
                ):
                    top_search.append(r)
            except Exception as error:
                if "202 Ratelimit" in str(error):
                    error_message = "Rate limit reached."
                else:
                    error_message = str(error)
                logger.warning(error_message)

        if len(top_search) == 0:
            return json.dumps({"error": error_message}, ensure_ascii=False)

        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = [self.fetch_async(session, r["href"]) for r in top_search]
            search_results = await asyncio.gather(*tasks)
            for r, sr in zip(top_search, search_results):
                if sr:
                    r["html"] = sr
        web_search_result = "\n\n".join([json.dumps(r) for r in top_search])
        return web_search_result

    def run(self, params: str) -> dict:
        logger = logging.getLogger(__name__)
        # Validate parameters
        if not self.validate(params=params):
            return json.dumps(
                {"error": "Invalid parameters for DuckDuckGoSearchAgent"},
                ensure_ascii=True,
            )
        # Load parameters
        params = json.loads(params)
        query = params.get("query", None)

        top_search = []
        error_message = "Unknown Error"
        with DDGS(headers=self.headers) as ddgs:
            try:
                for r in ddgs.text(
                    keywords=query,
                    region=self.region,
                    safesearch=self.safesearch,
                    max_results=self.num_results,
                ):
                    top_search.append(r)
            except Exception as error:
                if "202 Ratelimit" in str(error):
                    error_message = "Rate limit reached."
                else:
                    error_message = str(error)
                logger.warning(error_message)

        if len(top_search) == 0:
            return json.dumps({"error": error_message}, ensure_ascii=False)

        for r in top_search:
            page = self.fetch(url=r["href"])
            if page:
                r["html"] = page

        web_search_result = "\n\n".join([json.dumps(r) for r in top_search])
        return web_search_result

    async def fetch_async(self, session, url):
        try:
            await asyncio.sleep(self.pause)
            async with session.get(url) as response:
                data = await response.text()
                soup = BeautifulSoup(data, "html.parser")
                return self.remove_whitespaces(soup.find("body").text)
        except Exception as _:
            return None

    def fetch(self, url: str):
        try:
            time.sleep(self.pause)
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
