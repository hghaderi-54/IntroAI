"""Summarize a website using a local Ollama model."""
import argparse
import os
from openai import OpenAI
from main import build_messages
from scraper import fetch_website_contents

def summarize(url: str) -> str:
    client = OpenAI(base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama")
    response = client.chat.completions.create(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        messages=build_messages(fetch_website_contents(url)),
    )
    return response.choices[0].message.content or "No summary returned."

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Public HTTP or HTTPS website to summarize")
    print(summarize(parser.parse_args().url))

if __name__ == "__main__":
    main()
