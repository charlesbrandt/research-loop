import os
import requests
from typing import List, Type, Any
from pydantic import BaseModel, Field

from crewai.tools import BaseTool

class SearXNGSearchInput(BaseModel):
    query: str = Field(description="The search query.")

class SearXNGSearchTool(BaseTool):
    name: str = "SearXNG Search Tool"
    description: str = (
        "A tool that performs searches using a local SearXNG instance. "
        "Useful for general web searches when you need up-to-date information."
        "Input should be a string representing the search query."
    )
    args_schema: Type[BaseModel] = SearXNGSearchInput

    # It's good practice to make the base_url configurable
    searxng_base_url: str = Field(
        default_factory=lambda: os.getenv("SEARXNG_BASE_URL", "http://searxng:8080"),
        description="The base URL of the SearXNG instance.",
    )

    def _run(self, query: str) -> str:
        if not self.searxng_base_url:
            raise ValueError(
                "SEARXNG_BASE_URL is not set. "
                "Please set the SEARXNG_BASE_URL environment variable "
                "or pass it during tool initialization."
            )

        search_url = f"{self.searxng_base_url}/search"
        params = {
            "q": query,
            "format": "json",
        }
        try:
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            search_results = response.json()

            # Process the results to return a clean string
            formatted_results = []
            for result in search_results.get("results", []):
                title = result.get("title")
                url = result.get("url")
                content = result.get("content")
                if title and url:
                    formatted_results.append(f"Title: {title}\nURL: {url}\nContent: {content}\n---")
            
            if not formatted_results:
                return "No relevant search results found."

            return "\n".join(formatted_results)

        except requests.exceptions.RequestException as e:
            return f"An error occurred while connecting to SearXNG: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"
