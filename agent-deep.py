"""
A smolagent for deep-research tasks.

The agent can search the web, fetch and read full page contents, and synthesise
findings into a comprehensive, cited research summary — similar to a "deep
research" feature.

Usage:
    python agent-deep.py "What are the latest advances in photonic computing? Generate a concise report with key findings and URLs."
    python agent-deep.py --backend deepseek "Compare retrieval-augmented generation vs long-context models"
    python agent-deep.py --no-plan "Create a report on the current state of quantum machine learning, skipping the planning step and going straight to execution."
    python agent-deep.py --max-steps 25 --planning-interval 5 "How is CRISPR being used in agricultural biotechnology? Create a report with citations."
    python agent-deep.py --skill research "Summarise the latest advances in photonic computing."  # Load research skill
"""

import argparse
import asyncio
import sys
import warnings

from utils import load_memory_for_task, make_code_agent, setup_environment
from tools import (
    fetch_webpage,
    generate_research_report,
    read_memory,
    update_memory,
    web_search,
)

DEEP_RESEARCH_TOOLS = [
    web_search,
    fetch_webpage,
    generate_research_report,
    read_memory,
    update_memory,
]

DEEP_RESEARCH_IMPORTS = [
    "requests",
    "bs4",
    "ddgs",
    "markdownify",
    "json",
    "re",
    "datetime",
    "pathlib",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A smolagents deep-research agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=(
            "Research the topic 'recent advances in small language models "
            "(SLMs) under 3B parameters'.  Search the web for the latest "
            "information, fetch and read the most promising pages, then "
            "produce a concise, well-structured summary with key findings "
            "and citations (URLs)."
        ),
        help="The research question or topic for the agent to investigate.",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "deepseek"],
        default="openai",
        help="LLM backend to use (default: %(default)s).",
    )
    parser.add_argument(
        "--no-plan",
        action="store_true",
        help="Skip the planning step (run directly).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum number of steps the agent may take (default: %(default)s).",
    )
    parser.add_argument(
        "--planning-interval",
        type=int,
        default=1000,
        help="Re‑plan every N steps (default: %(default)s — plan only on step 1).",
    )
    parser.add_argument(
        "--skill",
        type=str,
        default=None,
        help="Name of a skill to load from skills/<name>/SKILL.md (e.g. 'research').",
    )
    args = parser.parse_args()

    # Load environment
    env = setup_environment()

    # Build the agent with deep-research tools
    runner = make_code_agent(
        env=env,
        backend=args.backend,
        with_planning=not args.no_plan,
        max_steps=args.max_steps,
        planning_interval=args.planning_interval,
        tools=DEEP_RESEARCH_TOOLS,
        additional_authorized_imports=DEEP_RESEARCH_IMPORTS,
        skill=args.skill,
    )
    print(f"🔍 Deep-research agent  |  backend: {args.backend}\n")

    # Run the agent on the task
    try:
        task_with_memory = load_memory_for_task(args.prompt)
        result = runner.run(task_with_memory)
        print(f"\n🎉 Research complete:\n{result}")
    except Exception as e:
        if "interrupted" in str(e).lower():
            print("\n🛑 Agent was interrupted by user.")
            sys.exit(0)
        raise
    finally:
        # Clean up asyncio event loop to avoid ResourceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.close()
            except Exception:
                pass
