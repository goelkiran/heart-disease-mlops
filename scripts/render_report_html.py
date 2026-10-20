"""
Renders the Markdown technical report into a publication-grade, self-contained HTML report
with embedded Base64 images and print-optimized CSS that fits cleanly in ~10-12 pages.
"""

from pathlib import Path
import base64
import re
import markdown

REPORT_DIR = Path(__file__).resolve().parent.parent / "report"
MD_PATH = REPORT_DIR / "MLOps_Assignment_1_Final_Report.md"
HTML_PATH = REPORT_DIR / "MLOps_Assignment_1_Final_Report.html"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLOps Assignment 01 Technical Report - Kiran Kumar (2025AE05377)</title>
    <style>
        :root {
            --primary: #1e3a8a;
            --secondary: #0284c7;
            --accent: #d97706;
            --text: #1f2937;
            --bg: #ffffff;
            --code-bg: #f8fafc;
            --border: #cbd5e1;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.45;
            color: var(--text);
            background-color: var(--bg);
            max-width: 880px;
            margin: 0 auto;
            padding: 24px;
            font-size: 13.5px;
        }

        h1, h2, h3, h4 {
            color: var(--primary);
            font-weight: 700;
            line-height: 1.25;
            break-after: avoid;
            page-break-after: avoid;
        }

        h1 {
            font-size: 1.65rem;
            border-bottom: 2.5px solid var(--secondary);
            padding-bottom: 8px;
            margin-top: 0;
            margin-bottom: 12px;
        }

        h2 {
            font-size: 1.25rem;
            border-bottom: 1.5px solid var(--border);
            padding-bottom: 4px;
            margin-top: 1.4em;
            margin-bottom: 8px;
        }

        h3 {
            font-size: 1.05rem;
            color: #1e293b;
            margin-top: 1.1em;
            margin-bottom: 6px;
        }

        p, li {
            font-size: 0.92rem;
            color: #334155;
            margin: 6px 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 0.82rem;
            break-inside: avoid;
            page-break-inside: avoid;
        }

        th, td {
            padding: 6px 10px;
            border: 1px solid var(--border);
            text-align: left;
        }

        th {
            background-color: #f1f5f9;
            color: var(--primary);
            font-weight: 600;
        }

        tr:nth-child(even) {
            background-color: #f8fafc;
        }

        code {
            font-family: "Cascadia Code", Consolas, "Courier New", monospace;
            background-color: #f1f5f9;
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 0.85em;
            color: #b91c1c;
        }

        pre {
            background-color: #0f172a;
            color: #f8fafc;
            padding: 10px 14px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.82rem;
            line-height: 1.4;
            margin: 10px 0;
            break-inside: avoid;
            page-break-inside: avoid;
        }

        pre code {
            background-color: transparent;
            color: inherit;
            padding: 0;
        }

        blockquote {
            border-left: 3.5px solid var(--secondary);
            margin: 10px 0;
            padding: 6px 12px;
            background-color: #f0f9ff;
            color: #0369a1;
            font-style: italic;
            font-size: 0.88rem;
        }

        hr {
            border: none;
            border-top: 1px solid var(--border);
            margin: 18px 0;
        }

        img {
            max-width: 82%;
            max-height: 240px;
            object-fit: contain;
            display: block;
            margin: 8px auto;
            border-radius: 4px;
            border: 1px solid var(--border);
            break-inside: avoid;
            page-break-inside: avoid;
        }

        em {
            display: block;
            text-align: center;
            font-size: 0.8rem;
            color: #64748b;
            margin-bottom: 10px;
        }

        a {
            color: var(--secondary);
            text-decoration: none;
            font-weight: 500;
        }

        a:hover {
            text-decoration: underline;
        }

        @media print {
            body {
                max-width: 100%;
                padding: 0;
                font-size: 9.5pt;
                line-height: 1.35;
            }
            h1 { font-size: 15pt; }
            h2 { font-size: 12pt; page-break-after: avoid; }
            h3 { font-size: 10.5pt; page-break-after: avoid; }
            p, li { font-size: 9pt; }
            table { font-size: 8pt; page-break-inside: avoid; }
            th, td { padding: 4px 6px; }
            pre { font-size: 7.5pt; padding: 6px 10px; page-break-inside: avoid; }
            img {
                max-height: 210px;
                max-width: 80%;
                page-break-inside: avoid;
            }
            @page {
                size: A4;
                margin: 12mm 12mm 12mm 12mm;
            }
        }
    </style>
</head>
<body>
    {content}
</body>
</html>
"""

def embed_images_as_base64(html_text: str) -> str:
    """Finds all <img src="..."> tags and replaces local file paths with Base64 data URIs."""
    def replacer(match):
        src = match.group(1)
        # Check if local relative path
        img_path = REPORT_DIR / src
        if img_path.exists():
            ext = img_path.suffix.lstrip(".").lower()
            if ext == "jpg":
                ext = "jpeg"
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f'src="data:image/{ext};base64,{b64}"'
        return match.group(0)

    return re.sub(r'src="([^"]+)"', replacer, html_text)


def render_report():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert Markdown to HTML with tables, code fences, and nl2br
    html_content = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br", "toc"]
    )

    # Embed Base64 images directly into the HTML
    html_content = embed_images_as_base64(html_content)

    full_html = HTML_TEMPLATE.replace("{content}", html_content)

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"[+] Successfully compiled report with Base64 embedded images to: {HTML_PATH}")


if __name__ == "__main__":
    render_report()
