---
name: research
description: "End-to-end deep-research pipeline: web search, page retrieval, and structured report generation."
version: 0.0.0
author: Nkluge-correa
license: MIT
---

# Deep-Research Pipeline

Search the web for information on a given topic, fetch and read full page contents, and synthesise findings into a comprehensive, well-formatted markdown research report with citations and embedded sources.

## Quick Reference

| Step               | Tool                       | Default Input / Behaviour           | Default Output                                |
|--------------------|----------------------------|-------------------------------------|-----------------------------------------------|
| 1. Web Search      | `web_search`               | Search query string                 | Numbered results with title, URL, and snippet |
| 2. Fetch Page      | `fetch_webpage`            | URL from a search result            | Full page title and body text as Markdown     |
| 3. Generate Report | `generate_research_report` | Topic string + synthesised findings | `reports/research_report_YYYY-MM-DD.md`       |

## Pipeline Overview

The deep-research pipeline follows three conceptual stages. In practice, the agent may loop over steps 1 and 2 multiple times — issuing several search queries and fetching many pages — before proceeding to step 3.

```
Research Topic
     │
     ▼
┌──────────────────┐
│ 1. Web Search    │  web_search(query, max_results)
│    (discover     │
│     sources)     │
└────────┬─────────┘
         │  (repeat for each promising source)
         ▼
┌──────────────────┐
│ 2. Fetch Page    │  fetch_webpage(url, max_chars)
│    (retrieve     │
│     content)     │
└────────┬─────────┘
         │  (once all findings are collected)
         ▼
┌──────────────────────────────┐
│ 3. Generate Research Report  │  generate_research_report(topic, findings)
│    (synthesised, structured) │
└──────────────────────────────┘
```

## Dependencies

| Package       | Purpose                                          |
|---------------|--------------------------------------------------|
| `requests`    | HTTP client for fetching web pages               |
| `beautifulsoup4` | HTML parsing and content extraction           |
| `markdownify` | Convert HTML to readable Markdown                |
| `ddgs`        | DuckDuckGo search API client                     |
| `re`          | URL extraction for the sources section           |

**Notes:**

- You will probably be working in a virtual environment (e.g., `venv` or `conda`) that already has these packages installed. If not, install them before running the pipeline.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install requests beautifulsoup4 markdownify ddgs
```

## 1. Web Search — `web_search`

Performs a web search using DuckDuckGo and returns formatted results. Each result includes a title, URL, and text snippet. Use this to discover sources and gather initial information on a topic. To get the full content of a promising result, follow up with `fetch_webpage`.

**Parameters:**

| Parameter     | Type  | Default | Description                                |
|---------------|-------|---------|--------------------------------------------|
| `query`       | `str` | —       | The search query string                    |
| `max_results` | `int` | `10`    | Maximum number of results to return (1–20) |

**Code:**

```python
def web_search(query: str, max_results: int = 10) -> str:
    from ddgs import DDGS

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=min(max_results, 20)))
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return f"No results found for '{query}'."

    lines = [f"Web search results for **'{query}'** ({len(results)} found):\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        href = r.get("href", "")
        body = r.get("body", "No snippet available")
        lines.append(f"{i}. **{title}**\n   URL: {href}\n   {body}\n")

    return "\n".join(lines)
```

**Notes:**

- Uses the `ddgs` library (install with `pip install ddgs` if not already available).
- The `DDGS` context manager handles session creation and teardown automatically.
- Results are capped at 20 internally regardless of `max_results`.
- Each result is numbered and includes a clickable URL and a short body snippet — sufficient to decide whether to fetch the full page.

## 2. Fetch Webpage — `fetch_webpage`

Fetches a webpage and returns its main text content as Markdown. The page is downloaded, parsed with BeautifulSoup, and converted to readable Markdown. Content is truncated if it exceeds `max_chars`.

**Parameters:**

| Parameter  | Type  | Default  | Description                                     |
|------------|-------|----------|-------------------------------------------------|
| `url`      | `str` | —        | The full URL of the webpage to fetch            |
| `max_chars`| `int` | `12000`  | Maximum characters to return                    |

**Code:**

```python
def fetch_webpage(url: str, max_chars: int = 12000) -> str:
    import requests
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Error fetching '{url}': {e}"

    # Determine encoding
    resp.encoding = resp.apparent_encoding or "utf-8"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Strip non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else "No title"

    # Try to extract the main content area, fall back to full body
    main = soup.find("main") or soup.find("article") or soup.body
    body_html = str(main) if main else str(soup)

    try:
        body_md = md(body_html, heading_style="ATX", strip=["img"])
    except Exception:
        body_md = body_html

    # Collapse whitespace
    import re
    body_md = re.sub(r"\n{3,}", "\n\n", body_md)
    body_md = re.sub(r"[ \t]+", " ", body_md)

    if len(body_md) > max_chars:
        body_md = body_md[:max_chars] + "\n\n... [content truncated] ..."

    return f"# {title}\n\nSource: {url}\n\n{body_md.strip()}"
```

**Notes:**

- Uses a desktop Chrome `User-Agent` header to avoid being blocked by common bot-detection filters.
- Strips non-content HTML elements (`script`, `style`, `nav`, `footer`, `header`, `aside`, `noscript`) before conversion so the output is just the readable text.
- Attempts to locate the page's main content area via `<main>` or `<article>` tags before falling back to the full `<body>` — this produces cleaner output for well-structured pages.
- Images are stripped from the Markdown output; only text content is preserved.
- The response encoding is auto-detected (`apparent_encoding`) to handle non-UTF-8 pages correctly.
- Content is truncated at `max_chars` (default 12,000 characters) to avoid overwhelming the agent's context window.

## 3. Generate Research Report — `generate_research_report`

Generates a structured, well-formatted markdown research report from the agent's synthesised findings. Use this at the **end** of a deep-research task to produce a final, human-readable report.

The report includes:
- Title and generation date.
- A **Executive Summary** section with the full synthesised findings.
- A **Sources & References** section listing all cited URLs (automatically extracted from the findings text).
- A **Methodology** section describing how the research was conducted.

**Parameters:**

| Parameter     | Type  | Default                                        | Description                            |
|---------------|-------|------------------------------------------------|----------------------------------------|
| `topic`       | `str` | —                                              | The research question or topic         |
| `findings`    | `str` | —                                              | The full synthesised research text     |
| `output_path` | `str` | `'reports/research_report_YYYY-MM-DD.md'`      | Where to save the report               |

**Code:**

```python
def generate_research_report(
    topic: str,
    findings: str,
    output_path: str = "",
) -> str:
    import re

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    # Default output path
    if not output_path:
        output_path = f"reports/research_report_{date_str}.md"

    # Extract all URLs from the findings text
    url_pattern = re.compile(r"https?://[^\s)>]+")
    urls = list(dict.fromkeys(url_pattern.findall(findings)))

    lines = [
        f"# Research Report: {topic}",
        "",
        f"**Generated:** {date_str} at {time_str}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        findings.strip(),
        "",
        "---",
        "",
        "## Sources & References",
        "",
    ]

    if urls:
        for i, url in enumerate(urls, 1):
            lines.append(f"{i}. [{url}]({url})")
    else:
        lines.append("_No URLs were explicitly cited in the findings._")

    lines += [
        "",
        "---",
        "",
        "## Methodology",
        "",
        "This report was generated by an automated deep-research agent using the "
        "following workflow:",
        "",
        "1. **Web search** — multiple targeted queries were executed via DuckDuckGo "
        "to discover relevant, high-quality sources.",
        "2. **Source retrieval** — the most promising pages were fetched in full and "
        "their main content extracted as markdown.",
        "3. **Synthesis** — the agent cross-referenced information across sources and "
        "produced a structured summary with inline citations.",
        "4. **Report generation** — this final markdown report was assembled "
        "automatically from the synthesised findings.",
        "",
        f"_Report auto-generated on {date_str}._",
    ]

    # Write the report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return (
        f"Research report saved to '{output_path}'.\n"
        f"  Topic: {topic}\n"
        f"  Sources cited: {len(urls)}"
    )
```

**Notes:**

- The `findings` text can contain arbitrary Markdown — headings, bullet points, inline code, and URLs. All formatting is preserved in the final report.
- URLs are automatically extracted from the findings with a regex and collected into the **Sources & References** section. Duplicate URLs are deduplicated while preserving order.
- If the findings text contains no URLs, the sources section displays a placeholder message instead of an empty list.
- The output directory (`reports/`) is created automatically if it does not exist.
- The report filename includes the current date, so multiple runs on different days will not overwrite each other.

## Notes

- The three tools are designed to be chained: `web_search` discovers sources, `fetch_webpage` retrieves their full content, and `generate_research_report` produces the final output. The agent can (and should) loop over search and fetch multiple times before generating the report.
- The `agent-deep.py` script orchestrates this pipeline using smolagents' `CodeAgent`, providing a configurable backend (OpenAI or DeepSeek), optional planning, and configurable max steps and planning intervals.
- Memory tools (`read_memory` and `update_memory`) are also available in the deep-research agent for persisting lessons across runs but are outside the scope of this skill document.
- For best results, the research topic should be specific and well-scoped. Broad topics (e.g., "AI") benefit from multiple targeted sub-queries rather than a single broad search.
