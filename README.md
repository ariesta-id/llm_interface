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