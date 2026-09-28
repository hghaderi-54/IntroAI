"""Fetch readable website text and links."""
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests

HEADERS = {"User-Agent": "WebsiteSummarizer/0.1"}

def _fetch_page(url):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a valid HTTP or HTTPS URL.")
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.content, "html.parser")

def fetch_website_contents(url):
    """Return the title and readable text, limited to 2,000 characters."""
    soup = _fetch_page(url)
    title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled page"
    body = soup.body or soup
    for element in body.find_all(["script", "style", "img", "input", "nav"]):
        element.decompose()
    text = body.get_text(separator="\n", strip=True)
    if not text:
        raise ValueError("The website did not contain readable text.")
    return (title + "\n\n" + text)[:2000]

def fetch_website_links(url):
    """Return nonempty link targets from a web page."""
    return [link['href'] for link in _fetch_page(url).find_all('a', href=True) if link['href']]
