"""
A smolagents agent for sales forecasting.

Usage:
    python agent-forecast.py                           # Uses OpenAI by default with the built-in prompt
    python agent-forecast.py --backend deepseek        # Use DeepSeek instead
    python agent-forecast.py --no-plan                 # Skip the planning step (go straight to execution)
    python agent-forecast.py --max-steps 30            # Allow more steps for complex tasks
    python agent-forecast.py --planning-interval 5     # Re-plan every 5 steps instead of only on step 1
    python agent-forecast.py --no-plan --max-steps 25  # Skip planning but allow more steps
    python agent-forecast.py --skill forecast          # Load the forecast skill instructions
"""

import argparse
import sys

from utils import load_memory_for_task, make_code_agent, setup_environment

from tools import (
    create_forecast_plot,
    download_dataset_from_hub,
    forecast_next_7_days,
    generate_final_report,
    preprocess_time_series_data,
    read_memory,
    train_xgboost_forecaster,
    update_memory,
)

FORECAST_TOOLS = [
    download_dataset_from_hub,
    preprocess_time_series_data,
    train_xgboost_forecaster,
    forecast_next_7_days,
    create_forecast_plot,
    generate_final_report,
    read_memory,
    update_memory,
]

FORECAST_IMPORTS = [
    "datasets",
    "pandas",
    "numpy",
    "pickle",
    "os",
    "shutil",
    "datetime",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A smolagents sales forecast agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Download the dataset 'AiresPucrs/time-series-data' from the "
        "Hugging Face Hub, preprocess it, train an XGBoost forecaster, "
        "predict the next 7 days of sales, create a plot, and generate "
        "a final report folder with all artifacts.",
        help="The task for the agent to perform.",
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
        help="Skip the planning step entirely (run CodeAgent directly).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum number of steps the agent may take (default: %(default)s).",
    )
    parser.add_argument(
        "--planning-interval",
        type=int,
        default=1000,
        help="Re-plan every N steps (default: %(default)s — plan only on step 1).",
    )
    parser.add_argument(
        "--skill",
        type=str,
        default=None,
        help="Name of a skill to load from skills/<name>/SKILL.md (e.g. 'forecast').",
    )
    args = parser.parse_args()

    # Load environment
    env = setup_environment()

    # Build the agent with forecast tools
    runner = make_code_agent(
        env=env,
        backend=args.backend,
        with_planning=not args.no_plan,
        max_steps=args.max_steps,
        planning_interval=args.planning_interval,
        tools=FORECAST_TOOLS,
        additional_authorized_imports=FORECAST_IMPORTS,
        skill=args.skill,
    )
    print(f"🤖 Using backend: {args.backend}\n")

    # Run the agent on the task
    try:
        # Always inject persistent memory into the task before running
        task_with_memory = load_memory_for_task(args.prompt)
        result = runner.run(task_with_memory)
        print(f"\n🎉 Agent result: {result}")
    except Exception as e:
        if "interrupted" in str(e).lower():
            print("\n🛑 Agent was interrupted by user.")
            sys.exit(0)
        raise
