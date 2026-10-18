"""
Renders the Markdown technical report into a standalone, publication-grade HTML report
with print-optimized styles, elegant typography, tables, and embedded figure references.
"""

from pathlib import Path
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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --primary: #1e3a8a;
            --secondary: #0284c7;
            --accent: #d97706;
            --text: #1f2937;
            --bg: #ffffff;
            --code-bg: #f3f4f6;
            --border: #e5e7eb;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.65;
            color: var(--text);
            background-color: var(--bg);
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 24px;
        }

        h1, h2, h3, h4 {
            color: var(--primary);
            font-weight: 700;
            line-height: 1.25;
            margin-top: 1.8em;
            margin-bottom: 0.6em;
        }

        h1 {
            font-size: 2.1rem;
            border-bottom: 2px solid var(--secondary);
            padding-bottom: 12px;
            margin-top: 0;
        }

        h2 {
            font-size: 1.5rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }

        h3 {
            font-size: 1.2rem;
            color: #1e293b;
        }

        p, li {
            font-size: 1rem;
            color: #374151;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.92rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        th, td {
            padding: 10px 14px;
            border: 1px solid var(--border);
            text-align: left;
        }

        th {
            background-color: #f8fafc;
            color: var(--primary);
            font-weight: 600;
        }

        tr:nth-child(even) {
            background-color: #f9fafb;
        }

        code {
            font-family: 'JetBrains Mono', monospace;
            background-color: var(--code-bg);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.88em;
            color: #b91c1c;
        }

        pre {
            background-color: #0f172a;
            color: #f8fafc;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        pre code {
            background-color: transparent;
            color: inherit;
            padding: 0;
        }

        blockquote {
            border-left: 4px solid var(--secondary);
            margin: 16px 0;
            padding: 8px 16px;
            background-color: #f0f9ff;
            color: #0369a1;
            font-style: italic;
        }

        hr {
            border: none;
            border-top: 1px solid var(--border);
            margin: 32px 0;
        }

        img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            margin: 16px 0;
            border: 1px solid var(--border);
        }

        @media print {
            body {
                max-width: 100%;
                padding: 0;
                font-size: 10.5pt;
            }
            h1 { font-size: 18pt; }
            h2 { font-size: 14pt; page-break-after: avoid; }
            h3 { font-size: 12pt; page-break-after: avoid; }
            table, pre, img { page-break-inside: avoid; }
            @page {
                margin: 1.8cm 1.5cm;
            }
        }
    </style>
</head>
<body>
    {content}
</body>
</html>
"""

def render_report():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert Markdown to HTML with tables, code fences, and toc extensions
    html_content = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br", "toc"]
    )

    full_html = HTML_TEMPLATE.replace("{content}", html_content)

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"[+] Successfully compiled report to: {HTML_PATH}")
    print("[*] Open this file in your browser and press 'Ctrl + P' -> 'Save as PDF' to generate the final PDF report.")


if __name__ == "__main__":
    render_report()
