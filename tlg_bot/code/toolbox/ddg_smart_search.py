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
from llm_agent_toolkit import Core

from mystorage import WebCache


class DDGSmartSearchTool(Tool):
    def __init__(
        self,
        llm: Core,
        web_cache: WebCache,
        safesearch: str = "off",
        region: str = "my-en",
        pause: float = 1.0,
        num_results: int = 5,
    ):
        Tool.__init__(self, DDGSmartSearchTool.function_info(), True)
        self.safesearch = safesearch
        self.region = region
        self.pause = pause
        self.num_results = num_results
        self.web_cache = web_cache
        self.llm = llm
        self.system_prompt = """Task: Summarize web page content, keep what is relevant to the {{Query}}.
        Ensure your answers are grounded.
        """

    @staticmethod
    def function_info():
        description = """
        Search for the relevant pages via DuckDuckGo API.
        Enhanced implementation:
        - has a LLM to extract the relevant sections.
        - cache results
        - Text only.
        """
        return FunctionInfo(
            name="DDGSmartSearchTool",
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

    async def run_async(self, params: str) -> str:
        logger = logging.getLogger(__name__)
        # Validate parameters
        if not self.validate(params=params):
            return json.dumps(
                {"error": "Invalid parameters for DDGSmartSearchTool"},
                ensure_ascii=True,
            )
        # Load parameters
        params = json.loads(params)
        query = params.get("query", None)

        top_search = []
        error_message = "Unknown Error"
        async with aiohttp.ClientSession(headers=self.headers) as session:
            with DDGS(headers=self.headers) as ddgs:
                try:
                    for _r in ddgs.text(
                        keywords=query,
                        region=self.region,
                        safesearch=self.safesearch,
                        max_results=self.num_results,
                    ):
                        challenge = f"{query}-{_r['href']}"
                        cached_content = self.web_cache.get(challenge)
                        if cached_content:
                            _r["html"] = cached_content
                            top_search.append(_r)
                            continue

                        page_html = await self.fetch_async(session, _r["href"])
                        if page_html is None:
                            top_search.append(_r)
                            continue

                        page_content_len = len(page_html)
                        if page_content_len < 1024:
                            summarized_content = page_html
                        else:
                            prompt = f"""Query: {query}
                            Here is the web search result:
                            ---
                            * Page Title: {_r['title']}
                            * Page Body: {_r['body']}
                            * Page Content: {page_html[:8_000]}
                            ---
                            """
                            try:
                                responses, usage = await self.llm.run_async(
                                    query=prompt,
                                    context=[
                                        {
                                            "role": "system",
                                            "content": self.system_prompt,
                                        }
                                    ],
                                )
                                summarized_content = responses[-1]["content"]
                                self.token_usage = self.token_usage + usage
                                # logger.info("$ Sumarized page content: %s", summarized_content)
                            except Exception as error:
                                logger.error(error, exc_info=True)
                                summarized_content = "Not Available"

                        # Store to cache
                        self.web_cache.set(challenge, summarized_content)
                        if summarized_content != "Not Available":
                            _r["html"] = summarized_content

                        top_search.append(_r)
                except Exception as error:
                    if "202 Ratelimit" in str(error):
                        error_message = "Rate limit reached."
                    else:
                        error_message = str(error)
                    logger.warning(error_message)

        if len(top_search) == 0:
            return json.dumps({"error": error_message}, ensure_ascii=False)

        web_search_result = "\n\n".join(json.dumps(top_search, ensure_ascii=False))
        return web_search_result

    def run(self, params: str) -> str:
        logger = logging.getLogger(__name__)
        # Validate parameters
        if not self.validate(params=params):
            return json.dumps(
                {"error": "Invalid parameters for DDGSmartSearchTool"},
                ensure_ascii=True,
            )
        # Load parameters
        params = json.loads(params)
        query = params.get("query", None)

        top_search = []
        error_message = "Unknown Error"
        with requests.Session() as session:
            with DDGS(headers=self.headers) as ddgs:
                try:
                    for _r in ddgs.text(
                        keywords=query,
                        region=self.region,
                        safesearch=self.safesearch,
                        max_results=self.num_results,
                    ):
                        challenge = f"{query}-{_r['href']}"
                        cached_content = self.web_cache.get(challenge)
                        if cached_content:
                            _r["html"] = cached_content
                            top_search.append(_r)
                            continue

                        page_html = self.fetch(session=session, url=_r["href"])
                        if page_html is None:
                            top_search.append(_r)
                            continue

                        page_content_len = len(page_html)
                        if page_content_len < 1024:
                            summarized_content = page_html
                        else:
                            prompt = f"""Query: {query}
                            Here is the web search result:
                            ---
                            * Page Title: {_r['title']}
                            * Page Body: {_r['body']}
                            * Page Content: {page_html[:8_000]}
                            ---
                            """
                            try:
                                responses, usage = self.llm.run(
                                    query=prompt,
                                    context=[
                                        {
                                            "role": "system",
                                            "content": self.system_prompt,
                                        }
                                    ],
                                )
                                summarized_content = responses[-1]["content"]
                                self.token_usage = self.token_usage + usage
                                # logger.info("$ Sumarized page content: %s", summarized_content)
                            except Exception as error:
                                logger.error(error, exc_info=True)
                                summarized_content = "Not Available"

                        # Store to cache
                        self.web_cache.set(challenge, summarized_content)
                        if summarized_content != "Not Available":
                            _r["html"] = summarized_content

                        top_search.append(_r)
                except Exception as error:
                    if "202 Ratelimit" in str(error):
                        error_message = "Rate limit reached."
                    else:
                        error_message = str(error)

        if len(top_search) == 0:
            return json.dumps({"error": error_message}, ensure_ascii=False)

        web_search_result = "\n\n".join(json.dumps(top_search, ensure_ascii=False))
        return web_search_result

    async def fetch_async(self, session: aiohttp.ClientSession, url: str) -> str | None:
        logger = logging.getLogger(__name__)
        try:
            cached_content: str | None = self.web_cache.get(url)
            if cached_content:
                logger.info("Load from cache: %s", url)
                return cached_content

            await asyncio.sleep(self.pause)
            async with session.get(url) as response:
                data = await response.text()
                soup = BeautifulSoup(data, "html.parser")
                output_string = self.remove_whitespaces(soup.find("body").text)
                # Store to cache
                self.web_cache.set(url, output_string)
                return output_string
        except Exception as _:
            return None

    def fetch(self, session: requests.Session, url: str) -> str | None:
        logger = logging.getLogger(__name__)
        try:
            cached_content: str | None = self.web_cache.get(url)
            if cached_content:
                logger.info("Load from cache: %s", url)
                return cached_content

            time.sleep(self.pause)
            page = session.get(url=url, headers=self.headers, timeout=2, stream=False)
            soup = BeautifulSoup(page.text, "html.parser")
            body = soup.find("body")
            if body:
                t = self.remove_whitespaces(body.text)
                # Store to cache
                self.web_cache.set(url, t)
                return t
            return None
        except Exception as _:
            return None
