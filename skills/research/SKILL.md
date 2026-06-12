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

The deep-research pipeline follows three conceptual stages. In practice, you should loop over steps 1 and 2 multiple times — issuing several search queries and fetching many pages — before proceeding to step 3.

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

| Package         | Purpose                               |
|-----------------|---------------------------------------|
| `requests`      | HTTP client for fetching web pages    |
| `beautifulsoup4`| HTML parsing and content extraction   |
| `markdownify`   | Convert HTML to readable Markdown     |
| `ddgs`          | DuckDuckGo search API client          |

These packages should already be installed in the virtual environment. If any are missing, install them with `pip install requests beautifulsoup4 markdownify ddgs`.

---

## 1. Web Search — `web_search`

Performs a web search via DuckDuckGo and returns formatted results (numbered, each with title, URL, and text snippet). Use this to discover sources and gauge which pages are worth a full fetch.

| Parameter     | Type  | Default | Description                                |
|---------------|-------|---------|--------------------------------------------|
| `query`       | `str` | —       | The search query string                    |
| `max_results` | `int` | `10`    | Maximum number of results to return (cap: 20) |

> **Tip:** For broad topics, issue multiple targeted sub-queries rather than one broad search. Each result includes enough context to decide whether to call `fetch_webpage` for the full content.

---

## 2. Fetch Webpage — `fetch_webpage`

Downloads a webpage and returns its main text content as clean Markdown. Non-content elements (`script`, `style`, `nav`, `footer`, `header`, `aside`, `noscript`) are stripped. The parser attempts to locate `<main>` or `<article>` content areas before falling back to the full `<body>`. Images are removed from the output.

| Parameter   | Type  | Default | Description                          |
|-------------|-------|---------|--------------------------------------|
| `url`       | `str` | —       | The full URL of the webpage to fetch |
| `max_chars` | `int` | `12000` | Maximum characters to return         |

> **Tip:** Content is truncated at `max_chars` to keep context manageable. If a page is too short or paywalled, try an alternative result from the search. Uses a desktop Chrome `User-Agent` to reduce bot-detection blocks. Encoding is auto-detected from the response.

---

## 3. Generate Research Report — `generate_research_report`

Produces a structured markdown report with four sections: title/date, **Executive Summary** (your synthesised findings), **Sources & References** (URLs auto-extracted from findings), and **Methodology** (describing the research workflow). Call this at the **end** of the research task.

| Parameter     | Type  | Default                                   | Description                        |
|---------------|-------|-------------------------------------------|------------------------------------|
| `topic`       | `str` | —                                         | The research question or topic     |
| `findings`    | `str` | —                                         | Full synthesised research text     |
| `output_path` | `str` | `'reports/research_report_YYYY-MM-DD.md'` | Where to save the report           |

> **Tip:** The `findings` text can contain arbitrary Markdown — headings, bullet points, inline code, and URLs. All formatting is preserved. URLs are automatically extracted and deduplicated for the Sources section. If no output path is given, the report is saved with today's date appended automatically.

---

## Execution Strategy

1. **Search broadly first** — start with 1–2 broad queries to map the landscape.
2. **Fetch the best** — evaluate snippets and fetch the 3–5 most promising pages.
3. **Search deeper if needed** — if gaps remain, issue more specific follow-up queries.
4. **Synthesise** — cross-reference findings across sources and write a structured summary with inline citations (URLs).
5. **Generate the report** — call `generate_research_report` with your topic and synthesised findings.

For best results, the research topic should be specific and well-scoped. The agent has access to memory tools (`read_memory`, `update_memory`) for persisting lessons across runs.
