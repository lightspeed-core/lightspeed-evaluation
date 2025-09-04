"""Response for evaluation."""

from typing import Optional

from httpx import Client
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate
from ols import config
from ols.constants import GenericLLMParameters
from ols.src.prompts.prompt_generator import GeneratePrompt

from .constants import REST_API_TIMEOUT
from .models import MODEL_OLS_PARAM, VANILLA_MODEL
from .prompts import BASIC_PROMPT
from .rag import retrieve_rag_chunks

set_debug(True)


# TODO: LCORE-271 pylint: disable=W0511
def get_model_response(
    query: str,
    provider: str,
    model: str,
    mode: str,
    api_client: Optional[Client] = None,
) -> str:
    """Get response depending upon the mode."""
    if mode == "ols":
        assert api_client is not None, "API client needs to be configured"
        response = api_client.post(
            "/v1/query",
            json={
                "query": query,
                "provider": provider,
                "model": model,
            },
            timeout=REST_API_TIMEOUT,
        )
        if response.status_code != 200:
            raise RuntimeError(response)
        return response.json()["response"].strip()

    prompt = PromptTemplate.from_template("{query}")
    prompt_input = {"query": query}
    provider_config = config.config.llm_providers.providers[provider]
    assert provider_config.type is not None, "Provider type needs to be specified in configuration"
    assert (
        provider_config.models is not None
    ), "Models for provider are not specified in configuration"
    model_config = provider_config.models[model]
    llm = VANILLA_MODEL[provider_config.type](
        model, provider_config
    ).load()  # pyright: ignore [reportCallIssue]

    if mode == "ols_param":
        max_resp_tokens = model_config.parameters.max_tokens_for_response
        override_params = {GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: max_resp_tokens}
        llm = MODEL_OLS_PARAM[provider_config.type](  # pyright: ignore [reportCallIssue]
            model, provider_config, override_params
        ).load()
    if mode == "ols_prompt":
        prompt, prompt_input = GeneratePrompt(query, [], []).generate_prompt(model)
    if mode == "ols_rag":
        rag_chunks = retrieve_rag_chunks(query, model, model_config)
        prompt, prompt_input = GeneratePrompt(query, rag_chunks, [], BASIC_PROMPT).generate_prompt(
            model
        )

    llm_chain = prompt | llm
    return llm_chain.invoke(input=prompt_input).content
