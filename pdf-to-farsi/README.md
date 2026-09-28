# PDF to Farsi

Extract text from an English PDF, translate it into Farsi in bounded chunks, and save UTF-8 text.

## Install

Use Python 3.13+. In this folder, run `uv sync`, or install `python -m pip install -r requirements.txt` in a virtual environment. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` for translation. `OPENAI_MODEL` defaults to `gpt-4.1-mini`.

## Run

```sh
uv run main.py examples/sample_story.pdf --extract-only
uv run main.py examples/sample_story.pdf --output translation.txt
```

The first command verifies extraction without an API key or model request. With pip, replace `uv run` with `python`. Omit `--output` to print the result. Existing output files are never overwritten.

## Longer documents

Text is split into chunks of up to 6,000 characters, preserving all content. Adjust with `--chunk-size 4000`. Each chunk makes a separate request. Chunk boundaries may reduce translation continuity, so review the output. Output is saved only after every chunk succeeds.

This tool handles text-based PDFs. Scanned PDFs need OCR first; encrypted PDFs may need to be unlocked. It produces plain text, not a translated PDF with the original layout. Extracted content is sent to OpenAI for translation and may incur charges. No translation quality guarantee is made.

Missing files, empty PDFs, invalid chunk sizes, output collisions, and API failures produce a nonzero exit status with an explanation. See [OpenAI error guidance](https://developers.openai.com/api/docs/guides/error-codes).
