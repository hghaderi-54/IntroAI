# AI Projects

Three focused projects: beginner machine learning, website summarization, and English-to-Farsi PDF translation.

## Choose a project

- **[Machine Learning](machine-learning/README.md)** — Titanic classification in English/Farsi and Tehran housing regression. Run in Colab, Kaggle, or Jupyter; no API key needed.
- **[Website Summarizer](website-summarizer/README.md)** — extract a web page and summarize it with OpenAI or Ollama. Includes Markdown export and an extraction-only mode.
- **[PDF to Farsi](pdf-to-farsi/README.md)** — extract PDF text, translate it in bounded chunks, and export UTF-8 text. Includes a local extraction-only mode.

## Quick start

For notebooks, follow the links above. For an application, enter its folder and run `uv sync`, then follow its README. Each application has its own dependency lockfile and `.env.example`.

## Verification

From this repository's root, install `python -m pip install -r requirements-dev.txt`, then run `python -m unittest discover -s tests -v`. Tests cover extraction, error handling, request formatting, chunking, and notebook structure without paid API calls. GitHub Actions runs these checks for pushes and pull requests.

API-backed generation requires your own credentials/model access or a running Ollama service. Automated tests use mock model responses and do not claim to assess translation or summarization quality.
