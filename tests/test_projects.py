import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nbformat
from pypdf import PdfWriter
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website-summarizer"))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

scraper = load("scraper_test", ROOT / "website-summarizer/scraper.py")
summary = load("summary_test", ROOT / "website-summarizer/main.py")
pdf = load("pdf_test", ROOT / "pdf-to-farsi/main.py")

class WebsiteTests(unittest.TestCase):
    def test_readable_text_and_timeout(self):
        response = Mock(content=b"<html><title>Example</title><body><nav>Menu</nav><p>Article</p><script>bad()</script></body></html>")
        with patch.object(scraper.requests, "get", return_value=response) as fetch:
            text = scraper.fetch_website_contents("https://example.com")
        self.assertIn("Article", text)
        self.assertNotIn("Menu", text)
        self.assertNotIn("bad()", text)
        self.assertEqual(fetch.call_args.kwargs["timeout"], 30)

    def test_invalid_url(self):
        with self.assertRaises(ValueError):
            scraper.fetch_website_contents("file:///local-file")

    def test_http_failure(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("404")
        with patch.object(scraper.requests, "get", return_value=response), self.assertRaises(requests.HTTPError):
            scraper.fetch_website_contents("https://example.com")

    def test_summary_request_and_empty_response(self):
        client = Mock()
        message = SimpleNamespace(content="A summary")
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        with patch.object(summary, "OpenAI", return_value=client), patch.object(summary, "load_dotenv"), patch.object(summary, "fetch_website_contents", return_value="Article"), patch.dict(os.environ, {"OPENAI_API_KEY": "mock-only"}):
            self.assertEqual(summary.summarize("https://example.com"), "A summary")
            self.assertTrue(client.chat.completions.create.call_args.kwargs["messages"][1]["content"].endswith("Article"))
            message.content = ""
            with self.assertRaises(ValueError):
                summary.summarize("https://example.com")

class PdfTests(unittest.TestCase):
    def test_extract_sample(self):
        self.assertTrue(pdf.pdf_to_text(str(ROOT / "pdf-to-farsi/examples/sample_story.pdf")))

    def test_blank_pdf(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blank.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=72, height=72)
            writer.write(str(path))
            with self.assertRaises(ValueError):
                pdf.pdf_to_text(str(path))

    def test_chunks_preserve_words(self):
        text = "word " * 150
        chunks = pdf.split_text(text, 100)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_long_word_and_invalid_size(self):
        text = "a" * 251
        self.assertEqual("".join(pdf.split_text(text, 100)), text)
        with self.assertRaises(ValueError):
            pdf.split_text("hello", 0)

    def test_translation_calls_every_chunk(self):
        text = "word " * 100
        with patch.object(pdf, "create_response", return_value="translation") as call:
            translated = pdf.translate_text(Mock(), text, 100)
        self.assertEqual(call.call_count, len(pdf.split_text(text, 100)))
        self.assertEqual(translated.count("translation"), call.call_count)

    def test_empty_translation_is_rejected(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
        with self.assertRaises(ValueError):
            pdf.create_response(client, pdf.create_messages("Hello"))

class NotebookTests(unittest.TestCase):
    def test_valid_notebooks_and_python_cells(self):
        paths = list((ROOT / "machine-learning/notebooks").rglob("*.ipynb"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    compile(cell.source, str(path), "exec")

if __name__ == "__main__":
    unittest.main()
