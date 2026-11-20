# LLM Interface
A boilerplate to interact with LLM API.

This is a personal repo. I recently often work with Fireworks API and `openai` library in several projects.

Run the example with `uv`.

```bash
uv run --env-file .env llm_interface.py
```

Calling the function from other python script requires some dependencies:
- `openai`
- `pandas`

If using `uv`:

```bash
uv add --script your_other_script.py openai pandas # once, to add inline script metadata
uv run --env-file .env your_other_script.py
```

The function will return a tuple:
- the LLM response
- input token (`prompt_token`)
- output token (`completion_token`)
- `first_token` predicted
- the logprob of the first token
- the probability (`np.exp(logprob)`) of the first token

The last three can be `None` and value only returned when `logprobs=True` is passed to the function.

Put the API key in `.env` following the example in `.envsample`.

## Non-streaming only

`stream=True` is not supported (it raises `ValueError`). Usage and cost metadata are read from the complete response.

## Metadata (tokens, cache, cost)

Besides `get_llm_response`, there are:

- `get_llm_response_with_meta(messages, **kwargs)` returns `(response_tuple, metadata)`.
- `get_llm_metadata(chat_completion)` returns the metadata dict for a raw response.

The dict has provider, generation id, input/cached/uncached/cache-write/completion/reasoning/output tokens, and cost (USD) total/input/output plus an estimated reasoning cost. See the `get_llm_metadata` docstring for every field.

The fields come automatically from the Hugging Face router (OpenRouter-style `usage` block) without extra request parameters. **Any field can be `None` if the provider doesn't report it** (e.g. caching differs per provider). Cache-hit vs cache-miss input cost is not available from the API.

## Images

Build a message with `user_message(text, *images)`. Each image is a local path (`.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`) or an `http(s)://` URL.

```python
from llm_interface import user_message, get_llm_response_with_meta

messages = [user_message("What's in this image?", "image-test.jpg")]  # or an https URL
(response, *_), meta = get_llm_response_with_meta(messages, max_tokens=4000)
```

- Local files are base64-encoded into a data URL. URLs are passed through; the provider fetches them.
- **Size guard:** images over 5 MB (`MAX_IMAGE_BYTES`) are blocked before any request is sent (`ImageTooLargeError`). Large images are untested and may be suboptimal for the model. To bypass: `user_message(text, image, allow_large=True)`.
- Local files are checked by bytes only (pixel dimensions are not checked). URL size is a best-effort `Content-Length` check and is skipped if the server doesn't report it.
- Image tokens count in `input_tokens`, so `get_llm_metadata` covers the cost.
