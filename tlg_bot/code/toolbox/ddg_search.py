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
from mystorage import WebCache


class DuckDuckGoSearchTool(Tool):
    """
    # DuckDuckGo Search Tool

    Search for the relevant pages via DuckDuckGo API.

    Notes:
    - Naive implementation
    - Text only
    """

    TRUNCATE_AFTER = 8192  # Pass to llm up to this length to limit token usage

    def __init__(
        self,
        web_cache: WebCache,
        safesearch: str = "off",
        region: str = "my-en",
        pause: float = 1.0,
        num_results: int = 5,
    ):
        Tool.__init__(self, DuckDuckGoSearchTool.function_info(), True)
        self.web_cache = web_cache
        self.safesearch = safesearch
        self.region = region
        self.pause = pause
        self.num_results = num_results

    @staticmethod
    def function_info():
        description = """
        Search for the relevant pages via DuckDuckGo API.
        Naive implementation:
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

    async def fetch_async(
        self,
        session: aiohttp.ClientSession,
        url: str,
        truncate_long_output: bool = True,
    ):
        """
        Fetch the content of a URL and store it in the web cache.

        If the URL is already cached, return the cached content.
        Otherwise, fetch the content using the Google Search API,
        process it, and store it in the web cache.

        Args:
            session (aiohttp.ClientSession): The session to use for the request.
            url (str): The URL to fetch.
            truncate_long_output (bool, optional): Whether to truncate the output if it is too long.
                Defaults to True.

        Returns:
            (str): The content of the URL, or "Page not available" if there was an error.
        """
        logger = logging.getLogger(__name__)
        cached_content: str | None = self.web_cache.get(url)
        if cached_content:
            return cached_content

        output_string, DEFAULT_OUTPUT = None, "Page not available"
        try:
            await asyncio.sleep(self.pause)
            async with session.get(url) as response:
                data = await response.text()
                soup = BeautifulSoup(data, "html.parser")
                output_string = self.remove_whitespaces(soup.find("body").text)
                if truncate_long_output:
                    output_string = output_string[: DuckDuckGoSearchTool.TRUNCATE_AFTER]
                return output_string
        except Exception as e:
            logger.error("Error fetching page %s: %s", url, e)
            return DEFAULT_OUTPUT
        finally:
            self.web_cache.set(url, output_string if output_string else DEFAULT_OUTPUT)

    def fetch(self, url: str, truncate_long_output: bool = True):
        """
        Fetch the content of a URL and store it in the web cache.

        If the URL is already cached, return the cached content.
        Otherwise, fetch the content using the Google Search API,
        process it, and store it in the web cache.

        Parameters:
            url (str): The URL to fetch.
            truncate_long_output (bool): Whether to truncate the output to a maximum length.
                Defaults to True.

        Returns:
            (str): The content of the URL, or "Page not available" if there was an error.
        """
        logger = logging.getLogger(__name__)
        cached_content: str | None = self.web_cache.get(url)
        if cached_content:
            return cached_content

        output_string, DEFAULT_OUTPUT = None, "Page not available"
        time.sleep(self.pause)
        try:
            page = requests.get(url=url, headers=self.headers, timeout=2, stream=False)
            soup = BeautifulSoup(page.text, "html.parser")
            body = soup.find("body")
            if body:
                t = body.text
                t = self.remove_whitespaces(t)
                output_string = t
                if truncate_long_output:
                    output_string = output_string[: DuckDuckGoSearchTool.TRUNCATE_AFTER]
                return output_string
            return DEFAULT_OUTPUT
        except Exception as e:
            logger.error("Error fetching page %s: %s", url, e)
            return DEFAULT_OUTPUT
        finally:
            self.web_cache.set(url, output_string if output_string else DEFAULT_OUTPUT)

    async def run_async(self, params: str) -> dict:
        logger = logging.getLogger(__name__)
        # Validate parameters
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
        query = params_dict.get("query", None)

        top_search = []
        async with aiohttp.ClientSession(headers=self.headers) as session:
            with DDGS(headers=self.headers) as ddgs:
                try:
                    for r in ddgs.text(
                        keywords=query,
                        region=self.region,
                        safesearch=self.safesearch,
                        max_results=self.num_results,
                    ):
                        page_html = await self.fetch_async(
                            session=session,
                            url=r["href"],
                        )
                        r["html"] = page_html
                        top_search.append(r)
                except Exception as error:
                    if "202 Ratelimit" in str(error):
                        error_message = "Rate limit reached."
                    else:
                        error_message = str(error)
                    logger.warning(error_message)
                    return json.dumps({"error": error_message}, ensure_ascii=False)

        if len(top_search) == 0:
            return json.dumps({"error": "No results"}, ensure_ascii=False)

        web_search_result = "\n\n".join(
            [json.dumps(r, ensure_ascii=False) for r in top_search]
        )
        return web_search_result

    def run(self, params: str) -> dict:
        logger = logging.getLogger(__name__)
        # Validate parameters
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
        query = params_dict.get("query", None)

        top_search = []
        with DDGS(headers=self.headers) as ddgs:
            try:
                for r in ddgs.text(
                    keywords=query,
                    region=self.region,
                    safesearch=self.safesearch,
                    max_results=self.num_results,
                ):
                    page = self.fetch(url=r["href"])
                    r["html"] = page
                    top_search.append(r)
            except Exception as error:
                if "202 Ratelimit" in str(error):
                    error_message = "Rate limit reached."
                else:
                    error_message = str(error)
                logger.warning(error_message)
                return json.dumps({"error": error_message}, ensure_ascii=False)

        if len(top_search) == 0:
            return json.dumps({"error": error_message}, ensure_ascii=False)

        web_search_result = "\n\n".join([json.dumps(r) for r in top_search])
        return web_search_result
