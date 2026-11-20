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
