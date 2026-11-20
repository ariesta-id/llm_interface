# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai",
#     "numpy"
# ]
# ///
from openai import OpenAI
import numpy as np
import base64
import mimetypes
import os
import urllib.request
import warnings

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
# MODEL = "accounts/fireworks/models/deepseek-v3-0324"
# MODEL = "accounts/fireworks/models/llama-v3p1-405b-instruct"
# MODEL = "accounts/fireworks/models/qwen2p5-72b-instruct"


def _request(messages, **kwargs):
    """Send a (non-streaming) chat completion request and return the raw response.

    Only `stream=False` is supported: usage/cost metadata is read from the
    complete response object.

    Raises:
    ValueError: If LLM_ environment variables are missing or `stream=True` is passed
    RuntimeError: For API connection errors
    """
    if kwargs.get("stream", False):
        raise ValueError("stream=True is not supported; this module is non-streaming only.")

    # Check for required CITA environment variables
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_API_URL")

    if not api_key:
        raise ValueError(
            "LLM_API_KEY environment variable is required. "
            "Get your API key from an OpenAI-compatible platform "
            "(like Fireworks.ai) and set it using:\n"
            "export LLM_API_KEY='your-key-here'"
        )
    if not base_url:
        raise ValueError(
            "LLM_API_URL environment variable is required. "
            "Set the API base URL using:\n"
            "export LLM_API_URL='https://api.fireworks.ai/inference/v1'"
        )

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client.chat.completions.create(
            model=MODEL, messages=messages, stream=False, **kwargs
        )
    except Exception as e:
        raise RuntimeError(f"API request failed: {str(e)}") from e


def _unpack_response(chat_completion, logprobs_requested):
    """Turn a raw response into the 6-tuple returned by `get_llm_response`."""
    message_content = chat_completion.choices[0].message.content
    prompt_token = chat_completion.usage.prompt_tokens
    completion_token = chat_completion.usage.completion_tokens
    if logprobs_requested:
        logprobs = chat_completion.choices[0].logprobs
        # Logprobs are optional: some providers/models don't return them.
        if logprobs is None or not logprobs.content:
            warnings.warn(
                "logprobs were requested but not returned by the API/provider; "
                "returning None for first_token, first_logprob and first_prob.",
                stacklevel=3,
            )
            return message_content, prompt_token, completion_token, None, None, None
        top_tokens = logprobs.content[0].top_logprobs
        first_token = top_tokens[0].token
        first_logprob = top_tokens[0].logprob
        first_prob = np.round(np.exp(first_logprob) * 100, 2)
        return (
            message_content,
            prompt_token,
            completion_token,
            first_token,
            first_logprob,
            first_prob,
        )
    return message_content, prompt_token, completion_token, None, None, None


def _dig(d, *path):
    """Safely read a nested value from dicts; return None if any level is missing."""
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def get_llm_metadata(chat_completion):
    """Extract usage, cache and cost metadata from a (non-streaming) chat completion.

    Works with the OpenRouter-style `usage` block returned by the Hugging Face
    router (https://router.huggingface.co/v1). These fields are returned
    automatically; no extra request parameter is needed.

    **Any field can be None** when the provider (or API) doesn't report it, e.g.
    `cached_input_tokens` is None/0 for providers without prompt caching. Derived
    fields are None if any of their inputs is None.

    Returns a dict with the keys:

    Identification
        provider (str): Provider that served the request, e.g. "DeepInfra".
        generation_id (str): Response id, e.g. "gen-...".
        model (str): Model id reported by the router.

    Tokens
        input_tokens (int): Prompt tokens (includes cached tokens).
        cached_input_tokens (int): Prompt tokens served from cache (cache hit).
        uncached_input_tokens (int): input_tokens - cached_input_tokens (cache miss).
        cache_write_tokens (int): Prompt tokens written to cache.
        completion_tokens (int): All generated tokens (reasoning + answer).
        reasoning_tokens (int): Reasoning/thinking tokens (part of completion_tokens).
        output_tokens (int): Answer tokens only: completion_tokens - reasoning_tokens.
        total_tokens (int): input_tokens + completion_tokens.

    Cost (USD, as billed)
        cost_total (float): Total cost of the request.
        cost_input (float): Cost of the input side (cache hit and miss combined;
            the API does not split them).
        cost_output (float): Cost of the completion side (reasoning + answer).
        cost_reasoning_estimate (float): ESTIMATE of the reasoning part of
            cost_output: cost_output * reasoning_tokens / completion_tokens.
            Reasoning is billed as output, so this is an apportionment, not a
            billed figure.

    Not available from the API: separate cost for cache-hit vs cache-miss input.
    """
    usage = chat_completion.usage.model_dump() if chat_completion.usage else {}
    extra = chat_completion.model_extra or {}

    input_tokens = _dig(usage, "prompt_tokens")
    cached = _dig(usage, "prompt_tokens_details", "cached_tokens")
    completion = _dig(usage, "completion_tokens")
    reasoning = _dig(usage, "completion_tokens_details", "reasoning_tokens")
    cost_output = _dig(usage, "cost_details", "upstream_inference_completions_cost")

    def both(a, b, fn):
        return fn(a, b) if a is not None and b is not None else None

    return {
        "provider": extra.get("provider"),
        "generation_id": chat_completion.id,
        "model": chat_completion.model,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "uncached_input_tokens": both(input_tokens, cached, lambda a, b: a - b),
        "cache_write_tokens": _dig(usage, "prompt_tokens_details", "cache_write_tokens"),
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "output_tokens": both(completion, reasoning, lambda a, b: a - b),
        "total_tokens": _dig(usage, "total_tokens"),
        "cost_total": _dig(usage, "cost"),
        "cost_input": _dig(usage, "cost_details", "upstream_inference_prompt_cost"),
        "cost_output": cost_output,
        "cost_reasoning_estimate": (
            cost_output * reasoning / completion
            if cost_output is not None and reasoning is not None and completion
            else None
        ),
    }


def get_llm_response(messages, **kwargs):
    """Get LLM response using `MODEL` via an OpenAI-compatible API (non-streaming).

    Returns a tuple: (message, prompt_token, completion_token, first_token,
    first_logprob, first_prob). The last three are None unless `logprobs=True`
    is passed and the provider returns logprobs (a warning is emitted if not).

    Raises:
    ValueError: If LLM_ environment variables are missing or `stream=True` is passed
    RuntimeError: For API connection errors
    """
    chat_completion = _request(messages, **kwargs)
    return _unpack_response(chat_completion, kwargs.get("logprobs", False))


def get_llm_response_with_meta(messages, **kwargs):
    """Like `get_llm_response`, but also returns the metadata dict.

    Returns `(response_tuple, metadata)` where `response_tuple` is exactly what
    `get_llm_response` returns and `metadata` is from `get_llm_metadata`
    (fields may be None if the provider doesn't report them).
    """
    chat_completion = _request(messages, **kwargs)
    return (
        _unpack_response(chat_completion, kwargs.get("logprobs", False)),
        get_llm_metadata(chat_completion),
    )


MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB; larger images are blocked unless allow_large=True
_IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class ImageTooLargeError(ValueError):
    """Raised when an image exceeds `MAX_IMAGE_BYTES` and `allow_large` is False."""


def _check_image_size(source, size_bytes, allow_large):
    if allow_large or size_bytes is None or size_bytes <= MAX_IMAGE_BYTES:
        return
    raise ImageTooLargeError(
        f"Image '{source}' is {size_bytes / 1024 / 1024:.1f} MB, above the "
        f"{MAX_IMAGE_BYTES / 1024 / 1024:.0f} MB limit. The request was NOT sent.\n"
        "WARNING: large images have not been tested with this model, and may not be "
        "optimal for the model's capability (higher cost, provider request limits, "
        "or worse results). Consider downscaling/compressing the image first.\n"
        "If you still want to send it, pass allow_large=True, e.g. "
        "user_message(text, image, allow_large=True)."
    )


def image_part(image, detail=None, allow_large=False):
    """Build an OpenAI-style `image_url` content part from a local path or URL.

    Args:
    image: Local file path (.jpg/.jpeg/.png/.webp/.gif) or an http(s):// URL.
        Local files are base64-encoded into a data URL. URLs are passed through
        as-is (the provider fetches them; nothing is downloaded here).
    detail: Optional "low" | "high" | "auto" hint. Providers may ignore it.
    allow_large: Bypass the size guard (`MAX_IMAGE_BYTES`). Large images are
        untested and may be suboptimal for the model.

    Size guard: local files are checked by byte size. For URLs a best-effort HTTP
    HEAD request reads `Content-Length`; if that is unavailable the URL is NOT
    size-checked. Image pixel dimensions are not checked.

    Raises:
    ImageTooLargeError: If the image is over `MAX_IMAGE_BYTES` and not allow_large.
    ValueError: For unsupported file types.
    FileNotFoundError: If a local path does not exist.
    """
    if str(image).startswith(("http://", "https://")):
        size = None
        try:
            req = urllib.request.Request(image, method="HEAD")
            with urllib.request.urlopen(req, timeout=5) as resp:
                length = resp.headers.get("Content-Length")
                size = int(length) if length else None
        except Exception:
            pass  # best effort only; the provider fetches the URL itself
        _check_image_size(image, size, allow_large)
        url = image
    else:
        ext = os.path.splitext(str(image))[1].lower()
        mime = _IMAGE_MIME_TYPES.get(ext) or mimetypes.guess_type(str(image))[0]
        if mime not in _IMAGE_MIME_TYPES.values():
            raise ValueError(
                f"Unsupported image type '{ext}'. "
                f"Supported: {', '.join(sorted(_IMAGE_MIME_TYPES))}"
            )
        _check_image_size(image, os.path.getsize(image), allow_large)
        with open(image, "rb") as f:
            url = f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

    image_url = {"url": url}
    if detail is not None:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def user_message(text, *images, detail=None, allow_large=False):
    """Build a user message with text and optional images.

    Example:
        messages = [user_message("What's in this image?", "image-test.jpg")]
        get_llm_response(messages)

    Args:
    text: The prompt text.
    *images: Local image paths and/or https URLs (see `image_part`).
    detail: Optional "low" | "high" | "auto" hint applied to every image.
    allow_large: Bypass the image size guard (untested; see `image_part`).
    """
    content = [{"type": "text", "text": text}]
    content += [image_part(i, detail=detail, allow_large=allow_large) for i in images]
    return {"role": "user", "content": content}


# Example usage
if __name__ == "__main__":
    prompt = "I know this doesn't make any sense. But please just answer 'yes' or 'no'."
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]

    (
        llm_message,
        prompt_token,
        completion_token,
        first_token,
        _,
        first_prob,
    ) = get_llm_response(
        messages,
        logprobs=True,
        top_logprobs=2,
        temperature=0,
    )
    print(f"""LLM message: {llm_message}
Prompt tokens: {prompt_token}
Completion tokens: {completion_token}
First token: {first_token}
First probability: {first_prob}
""")
    response = get_llm_response(
        messages,
    )
    print("LLM message: ", response[0])
    _, meta = get_llm_response_with_meta(messages)
    print("\nMetadata (fields may be None if the provider doesn't report them):")
    for key, value in meta.items():
        print(f"  {key}: {value}")
