# PDF to Farsi

A command-line tool that extracts text from an English PDF and translates it into Farsi using OpenAI.

## Setup

Use Python 3.13 or later. Run `uv sync`, or create a virtual environment and run `python -m pip install -r requirements.txt`. Copy `.env.example` to `.env` and set your own `OPENAI_API_KEY`.

## Run

```sh
uv run main.py examples/sample_story.pdf --output translation.txt
```

With pip, use `python main.py` instead of `uv run main.py`. Omit `--output` to print the translation. `OPENAI_MODEL` is optional and defaults to `gpt-4.1-mini`.

This tool supports text-based PDFs, not OCR or PDF layout reconstruction. It sends the extracted text to OpenAI in one request, so use short documents that fit the model's limits. API requests may incur charges. Translation is saved as UTF-8 plain text; review it before relying on it.
