import os
from dotenv import load_dotenv
from scraper import fetch_website_contents
from IPython.display import Markdown, display
from openai import OpenAI


# -------------------------
# Environment Setup
# -------------------------

load_dotenv(override=True)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

if API_KEY != API_KEY.strip():
    raise ValueError("OPENAI_API_KEY contains leading or trailing whitespace.")

client = OpenAI(api_key=API_KEY)


# -------------------------
# Prompts
# -------------------------

SYSTEM_PROMPT = """
You are a snarky assistant that analyzes the contents of a website
and provides a short, snarky, humorous summary, ignoring navigation-related text.
Respond in markdown. Do not wrap markdown in a code block.
"""

USER_PROMPT_PREFIX = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, summarize those too.
"""


def build_messages(website_text: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": f"{USER_PROMPT_PREFIX.strip()}\n\n{website_text}"}
    ]


# -------------------------
# Core Logic
# -------------------------

def summarize(url: str) -> str:
    website_text = fetch_website_contents(url)

    if not website_text:
        raise ValueError("Failed to fetch website content.")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=build_messages(website_text)
    )

    return response.choices[0].message.content


def display_summary(url):
    summary = summarize(url)
    print("\n--- Summary ---\n")
    print(summary)



# -------------------------
# Entry Point
# -------------------------

def main():
    url = "https://edwarddonner.com"
    display_summary(url)


if __name__ == "__main__":
    main()
