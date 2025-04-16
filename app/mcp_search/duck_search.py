from duckduckgo_search import DDGS
from typing import List
from pydantic import BaseModel, Field


class DuckDuckGoSearchResult(BaseModel):
    """
    Represents a single search result from DuckDuckGo.
    """
    title: str = Field(..., description="The title of the search result.")
    link: str = Field(..., description="The URL of the search result.")
    snippet: str = Field(..., description="A short description or snippet of the search result.")


class DuckDuckGoSearchResults(BaseModel):
    """
    Represents a collection of search results from DuckDuckGo.
    """
    results: List[DuckDuckGoSearchResult] = Field(..., description="A list of DuckDuckGo search results.")


class DuckDuckGoSearcher:
    """
    Performs searches on DuckDuckGo using the duckduckgo_search library.
    """
    def search(self, query: str, num_results: int = 10) -> DuckDuckGoSearchResults:
        """
        Performs a search on DuckDuckGo and returns the results.

        Args:
            query: The search query string.
            num_results: The number of search results to return (default: 10).

        Returns:
            A DuckDuckGoSearchResults object containing a list of search results.
            Returns an empty list if no results are found or if an error occurs.
        """
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, region='de-de', max_results=num_results)]

            formatted_results = []
            for result in results:
                formatted_results.append(
                    DuckDuckGoSearchResult(
                        title=result.get("title", "No title provided"),
                        link=result.get("href", "No link provided"),
                        snippet=result.get("body", "No description available"),
                    )
                )

            return DuckDuckGoSearchResults(results=formatted_results)

        except Exception as e:
            print(f"An error occurred during the search: {e}")
            return DuckDuckGoSearchResults(results=[])


# Example usage:
if __name__ == "__main__":
    search_query = "Was ist AI?"
    searcher = DuckDuckGoSearcher()
    search_results: DuckDuckGoSearchResults = searcher.search(search_query)

    print(search_results)

    if search_results.results:
        print(f"Search results for '{search_query}':")
        for i, result in enumerate(search_results.results):
            print(f"\nResult {i+1}:")
            print(f"  Title: {result.title}")
            print(f"  Link: {result.link}")
            print(f"  Snippet: {result.snippet}")
    else:
        print(f"No results found for '{search_query}' or an error occurred.")