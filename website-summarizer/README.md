# Website Summarizer

A command-line tool for extracting and summarizing static web pages. Choose OpenAI or a local Ollama model.

## Install

Use Python 3.13+. In this folder, run `uv sync`. Alternatively, create a virtual environment and run `python -m pip install -r requirements.txt`.

For OpenAI, copy `.env.example` to `.env` and set `OPENAI_API_KEY`. `OPENAI_MODEL` defaults to `gpt-4.1-mini` and can be changed in `.env`. Never commit credentials.

## Run

```sh
uv run main.py https://example.com --extract-only
uv run main.py https://example.com --output summary.md
```

The first command makes no model/API request. Output paths must not already exist. With pip, replace `uv run` with `python`.

For Ollama, start the local service, run `ollama pull llama3.2`, then:

```sh
uv run ollama_summarizer.py https://example.com
```

The Ollama variant needs no OpenAI API key. Override `OLLAMA_MODEL` and `OLLAMA_BASE_URL` in the shell environment when needed.

## Behavior and limits

The scraper removes scripts, styles, and navigation, checks HTTP failures, and uses a 30-second fetch timeout. It sends at most 2,000 characters to the model. It does not execute page JavaScript, bypass logins, or summarize an entire large website. OpenAI requests send the extracted text to OpenAI and may incur charges.

Errors are reported with a nonzero exit status. Empty model responses are rejected. For API failures, check credentials, connectivity, quota, and model access. See [OpenAI error guidance](https://developers.openai.com/api/docs/guides/error-codes).
