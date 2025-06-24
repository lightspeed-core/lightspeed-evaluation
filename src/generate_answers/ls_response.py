"""LightSpeed client module."""

import hashlib
import json
import logging
from typing import cast

from diskcache import Cache
from httpx import Client

logger = logging.getLogger(__name__)


class LSClient:  # pylint: disable=too-few-public-methods
    """LightSpeed client."""

    def __init__(
        self, ls_url: str, provider: str, model: str, cache_dir: str = "./llm_cache"
    ):
        """Init LightSpeed."""
        self.url = ls_url
        self.provider = provider
        self.model = model
        self.client = Client(base_url=ls_url, verify=False)

        # Timeout in seconds
        self.rest_api_timeout = 60

        # Disk cache for caching LLM responses
        self.cache = Cache(cache_dir)

    def _get_cache_key(self, query: str) -> str:
        """Get cache key for the query."""
        return hashlib.sha256(
            f"{self.url}-{self.provider}-{self.model}: {query}".encode()
        ).hexdigest()

    def _add_answer_to_cache(self, query: str, answer: str) -> None:
        """Add answer to disk cache."""
        key = self._get_cache_key(query)
        self.cache[key] = answer

    def _get_cached_answer(self, query: str) -> str | None:
        """Get answer from the disk cache."""
        key = self._get_cache_key(query)
        return cast(str | None, self.cache.get(key))

    def get_answer(self, query: str, skip_cache: bool = False) -> str:
        """Get LLM answer for query."""
        if not skip_cache:
            cached_answer = self._get_cached_answer(query)
            if cached_answer is not None:
                logging.info("Returning cached answer for query '%s'", query)
                return cached_answer

        logging.info("Calling LightSpeed service for query '%s'", query)
        response = self.client.post(
            "/v1/query",
            json={
                "query": query,
                "provider": self.provider,
                "model": self.model,
            },
            timeout=self.rest_api_timeout,
        )

        if response.status_code != 200:
            logger.error(
                "Status: %d, query='%s', response='%s'",
                response.status_code,
                query,
                json.dumps(response.json()),
            )
            raise RuntimeError(response)

        answer = response.json()["response"].strip()
        if not answer:
            logger.error(
                "Empty answer: query='%s', response='%s'",
                query,
                json.dumps(response.json()),
            )
            raise RuntimeError(response)

        self._add_answer_to_cache(query, answer)
        return answer
