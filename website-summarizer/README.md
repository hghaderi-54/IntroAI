# Website Summarizer

Summarize readable website content from the command line using OpenAI or a local Ollama model.

## Setup

Use Python 3.13 or later. Run `uv sync`, or create a virtual environment and run `python -m pip install -r requirements.txt`.

Copy `.env.example` to `.env` and set your own `OPENAI_API_KEY` for the OpenAI option. Never commit your key. `OPENAI_MODEL` is optional and defaults to `gpt-4.1-mini`.

## Run

```sh
uv run main.py https://example.com
```

For local Ollama, start Ollama and download a model with `ollama pull llama3.2`, then run:

```sh
uv run ollama_summarizer.py https://example.com
```

With pip, use `python` instead of `uv run`. The Ollama option does not require an OpenAI API key. Override `OLLAMA_MODEL` and `OLLAMA_BASE_URL` using environment variables if needed.

The scraper reads static HTML, checks HTTP errors, uses a 30-second timeout, and limits text to 2,000 characters. JavaScript-rendered content and paywalled pages may not be available. OpenAI requests send the extracted text to OpenAI and may incur API charges.
