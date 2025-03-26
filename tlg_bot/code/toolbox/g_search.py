"""
Module DocString
"""

import os
import re
import logging
import random
import json
import asyncio
import time
import aiohttp
import requests
from bs4 import BeautifulSoup
from llm_agent_toolkit import (
    Tool,
    FunctionInfo,
    FunctionProperty,
    FunctionPropertyType,
    FunctionParameters,
)
from mystorage import WebCache

logger = logging.getLogger(__name__)


class GoogleSearchTool(Tool):
    """
    # Google Search Tool

    Search for the relevant pages via Google API.

    Notes:
    - Naive implementation
    - Text only

    Warning:
    - Pricing - (https://developers.google.com/custom-search/docs/paid_element#pricing)
    - Remember to add `GOOGLE_SEARCH_API` and `CUSTOM_SEARCH_ID` to .env
    """

    TRUNCATE_AFTER = 8192  # Pass to llm up to this length to limit token usage

    def __init__(
        self,
        web_cache: WebCache,
        url: str = "https://www.googleapis.com/customsearch/v1",
        pause: float = 1.0,
        num_results: int = 5,
    ):
        """
        Initialize the GoogleSearchTool.

        Args:
            web_cache (WebCache):
                A cache object to store web page content and query responses.
            url (str):
                The URL to use for the Google Custom Search API. Defaults to "https://www.googleapis.com/customsearch/v1".
            pause (float):
                The pause time between requests. Defaults to 1.0.
            num_results (int):
                The number of search results to return. Defaults to 5.

        Returns:
            None
        """
        Tool.__init__(self, GoogleSearchTool.function_info(), True)
        self.web_cache = web_cache
        self.url = url
        self.pause = pause
        self.num_results = num_results

    @staticmethod
    def function_info():
        """
        Generates a FunctionInfo object for the GoogleSearchTool.

        This static method creates a FunctionInfo instance that describes
        the GoogleSearchTool's capabilities for querying a specific knowledge
        base given by the title. It includes the tool's name, description,
        and the required parameters.

        Parameters:
            None

        Returns:
            A FunctionInfo object containing the tool's metadata and expected
            input parameters.
        """
        description = """
        Search for the relevant pages via Google API.
        Naive implementation:
        - Text only.
        """
        return FunctionInfo(
            name="GoogleSearchTool",
            description=description,
            parameters=FunctionParameters(
                properties=[
                    FunctionProperty(
                        name="query",
                        type=FunctionPropertyType.STRING,
                        description="The query keyword. Please avoid vague and short keyword.",
                    ),
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

    @staticmethod
    def create_params(query: str, num_results: int) -> dict:
        """
        Create the parameters dictionary for the Google Search API.

        This method takes a search query and the number of results to return,
        and returns a dictionary of parameters for the Google Search API.

        Parameters:
            query (str): The search query.
            num_results (int): The number of results to return.

        Returns:
            A dictionary of parameters for the Google Search API.
        """
        return {
            "key": os.environ["GOOGLE_SEARCH_API"],
            "cx": os.environ["CUSTOM_SEARCH_ID"],
            "q": query,
            "num": num_results,
        }

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
                    output_string = output_string[: GoogleSearchTool.TRUNCATE_AFTER]
                return output_string
            return DEFAULT_OUTPUT
        except Exception as e:
            logger.error("Error fetching page %s: %s", url, e)
            return DEFAULT_OUTPUT
        finally:
            self.web_cache.set(url, output_string if output_string else DEFAULT_OUTPUT)

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
                    output_string = output_string[: GoogleSearchTool.TRUNCATE_AFTER]
                return output_string
        except Exception as e:
            logger.error("Error fetching page %s: %s", url, e)
            return DEFAULT_OUTPUT
        finally:
            self.web_cache.set(url, output_string if output_string else DEFAULT_OUTPUT)

    def run(self, params: str) -> str:
        """
        Execute a search query using the Google Search API.

        This method takes a JSON string of search parameters, validates them,
        and performs a search query using the Google Search API. It retrieves
        search results and fetches the HTML content of each result page.

        Parameters:
            params (str): JSON string containing the search parameters, including
                        the 'query' keyword.

        Returns:
            str: A JSON string containing the search results or an error message
                if the parameters are invalid or if an error occurs during the
                search process.
        """
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
        gs_params = self.create_params(query=query, num_results=self.num_results)
        output = []
        try:
            response = requests.get(self.url, params=gs_params, timeout=None)
            response.raise_for_status()
            response_obj = response.json()
            # Check if the response is valid
            if "items" not in response_obj:
                return json.dumps({"error": "No results"}, ensure_ascii=False)
            # Get the search results
            items: list = response_obj["items"]
            for item in items:
                _item = {
                    "title": item["title"],
                    "link": item["link"],
                    "snippet": item["snippet"],
                    "htmlSnippet": item["htmlSnippet"],
                }
                _item["html"] = self.fetch(url=item["link"])
                output.append(_item)
            # Concatenate the search results
            output_string = "\n\n".join(
                [json.dumps(o, ensure_ascii=False) for o in output]
            )
            return output_string
        except aiohttp.ClientError as http_err:
            error_message = str(http_err)
            return json.dumps({"error": error_message}, ensure_ascii=False)
        except Exception as e:
            error_message = str(e)
            return json.dumps({"error": error_message}, ensure_ascii=False)

    async def run_async(self, params: str) -> str:
        """
        Execute a search query using the Google Search API.

        This method takes a JSON string of search parameters, validates them,
        and performs a search query using the Google Search API. It retrieves
        search results and fetches the HTML content of each result page.

        Parameters:
            params (str): JSON string containing the search parameters, including
                        the 'query' keyword.

        Returns:
            str: A JSON string containing the search results or an error message
                if the parameters are invalid or if an error occurs during the
                search process.
        """
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
        gs_params = self.create_params(query=query, num_results=self.num_results)
        output = []
        try:
            async with aiohttp.ClientSession() as google_session:
                async with aiohttp.ClientSession(headers=self.headers) as page_session:
                    async with google_session.get(
                        self.url, params=gs_params
                    ) as response:
                        response.raise_for_status()
                        response_obj = await response.json()
                        # Check if the response is valid
                        if "items" not in response_obj:
                            return json.dumps(
                                {"error": "No results"}, ensure_ascii=False
                            )
                        # Get the search results
                        for item in response_obj["items"]:
                            _item = {
                                "title": item["title"],
                                "link": item["link"],
                                "snippet": item["snippet"],
                                "htmlSnippet": item["htmlSnippet"],
                            }
                            _item["html"] = await self.fetch_async(
                                session=page_session, url=item["link"]
                            )
                            output.append(_item)
            # Concatenate the search results
            output_string = "\n\n".join(
                [json.dumps(o, ensure_ascii=False) for o in output]
            )
            return output_string
        except aiohttp.ClientError as http_err:
            error_message = str(http_err)
            return json.dumps({"error": error_message}, ensure_ascii=False)
        except Exception as e:
            error_message = str(e)
            return json.dumps({"error": error_message}, ensure_ascii=False)
