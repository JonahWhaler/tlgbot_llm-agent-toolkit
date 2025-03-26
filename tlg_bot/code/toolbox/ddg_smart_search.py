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
    """
    # DuckDuckGo Smart Search Tool

    Search for the relevant pages via DuckDuckGo API.

    Enhanced implementation:
    - Uses LLM to extract the relevant sections.
    - Text only.
    """

    MIN_PAGE_LENGTH = 1024  # Skip llm if page is too short
    TRUNCATE_AFTER = 8192  # Pass to llm up to this length to limit token usage

    def __init__(
        self,
        llm: Core,
        web_cache: WebCache,
        safesearch: str = "off",
        region: str = "my-en",
        pause: float = 1.0,
        num_results: int = 5,
    ):
        """
        Initialize the DDGSmartSearchTool.

        Args:
            llm (Core): An instance of the LLM core used for processing queries.
            web_cache (WebCache): A cache object to store web page content and query responses.
            safesearch (str, optional): The safesearch setting for the search. Defaults to "off".
            region (str, optional): The region for the search. Defaults to "my-en".
            pause (float, optional): The time to pause between requests. Defaults to 1.0 seconds.
            num_results (int, optional): The number of search results to return. Defaults to 5.

        Sets up the necessary attributes for performing smart searches and caching results.
        """
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
        """
        Generates a FunctionInfo object for the DDGSmartSearchTool.

        This static method creates a FunctionInfo instance that describes
        the DDGSmartSearchTool's capabilities for querying a specific knowledge
        base given by the title. It includes the tool's name, description,
        and the required parameters.

        Parameters:
            None

        Returns:
            A FunctionInfo object containing the tool's metadata and expected
            input parameters.
        """
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
                    output_string = output_string[: DDGSmartSearchTool.TRUNCATE_AFTER]
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
                    output_string = output_string[: DDGSmartSearchTool.TRUNCATE_AFTER]
                return output_string
            return DEFAULT_OUTPUT
        except Exception as e:
            logger.error("Error fetching page %s: %s", url, e)
            return DEFAULT_OUTPUT
        finally:
            self.web_cache.set(url, output_string if output_string else DEFAULT_OUTPUT)

    async def run_async(self, params: str) -> str:
        """
        Execute a search query using the DuckDuckGo Search API.

        This method takes a JSON string of search parameters, validates them,
        and performs a search query using the DuckDuckGo Search API. It retrieves
        search results and fetches the HTML content of each result page.

        It then uses the LLM to extract the relevant sections from the HTML content.

        Parameters:
            params (str): JSON string containing the search parameters, including
                the 'query' keyword.

        Returns:
            str: A JSON string containing the search results or an error message
                if the parameters are invalid or if an error occurs during the
                search process.
        """
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
                        if len(page_html) < DDGSmartSearchTool.MIN_PAGE_LENGTH:
                            summarized_content = page_html
                        else:
                            prompt = f"""Query: {query}
                            Here is the web search result:
                            ---
                            * Page Title: {_r['title']}
                            * Page Body: {_r['body']}
                            * Page Content: {page_html}
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
                                # Update token usage
                                self.token_usage = self.token_usage + usage
                                # Store to cache
                                self.web_cache.set(challenge, summarized_content)
                            except Exception as e:
                                logger.error(
                                    "Error processing page: %s", e, exc_info=True
                                )
                                summarized_content = page_html[
                                    : DDGSmartSearchTool.MIN_PAGE_LENGTH
                                ]
                        _r["html"] = summarized_content
                        top_search.append(_r)
                except Exception as error:
                    if "202 Ratelimit" in str(error):
                        error_message = "Rate limit reached."
                    else:
                        error_message = str(error)
                    logger.warning(error_message)
                    return json.dumps({"error": error_message}, ensure_ascii=False)

        if len(top_search) == 0:
            return json.dumps({"error": "No results"}, ensure_ascii=False)

        web_search_result = "\n\n".join(json.dumps(top_search, ensure_ascii=False))
        return web_search_result

    def run(self, params: str) -> str:
        """
        Execute a search query using the DuckDuckGo search API.

        This method takes a JSON string of search parameters, validates them,
        and performs a search query using the DuckDuckGo search API. It retrieves
        search results and fetches the HTML content of each result page.

        It then uses the LLM to extract the relevant sections from the HTML content.

        Parameters:
            params (str): JSON string containing the search parameters, including
                the 'query' keyword.

        Returns:
            str: A JSON string containing the search results or an error message
                if the parameters are invalid or if an error occurs during the
                search process.
        """
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
        error_message = "Unknown Error"
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

                    page_html = self.fetch(url=_r["href"])
                    if len(page_html) < DDGSmartSearchTool.MIN_PAGE_LENGTH:
                        summarized_content = page_html
                    else:
                        prompt = f"""Query: {query}
                        Here is the web search result:
                        ---
                        * Page Title: {_r['title']}
                        * Page Body: {_r['body']}
                        * Page Content: {page_html}
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
                            # Update token usage
                            self.token_usage = self.token_usage + usage
                            # Store to cache
                            self.web_cache.set(challenge, summarized_content)
                        except Exception as e:
                            logger.error("Error processing page: %s", e, exc_info=True)
                            summarized_content = page_html[
                                : DDGSmartSearchTool.MIN_PAGE_LENGTH
                            ]
                    _r["html"] = summarized_content
                    top_search.append(_r)
            except Exception as error:
                if "202 Ratelimit" in str(error):
                    error_message = "Rate limit reached."
                else:
                    error_message = str(error)
                logger.warning(error_message)
                return json.dumps({"error": error_message}, ensure_ascii=False)

        if len(top_search) == 0:
            return json.dumps({"error": "No results"}, ensure_ascii=False)

        web_search_result = "\n\n".join(json.dumps(top_search, ensure_ascii=False))
        return web_search_result
