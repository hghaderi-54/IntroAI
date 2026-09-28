"""Summarize a website using a local Ollama model."""
import argparse
import os
from openai import OpenAI, OpenAIError
from requests import RequestException
from main import build_messages
from scraper import fetch_website_contents

def summarize(url: str) -> str:
    client = OpenAI(base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama", timeout=120, max_retries=0)
    response = client.chat.completions.create(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        messages=build_messages(fetch_website_contents(url)),
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("The local model returned an empty summary.")
    return content.strip()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Public HTTP or HTTPS website to summarize")
    args = parser.parse_args()
    try:
        print(summarize(args.url))
    except (ValueError, RequestException) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except OpenAIError:
        parser.exit(1, "Ollama request failed. Start Ollama and ensure the selected model is installed.\n")

if __name__ == "__main__":
    main()
