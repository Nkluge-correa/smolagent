"""
A simple smolagents agent using DeepSeek as the backend.

Usage:
    python agent.py "I need a report for the time-series data at 'AiresPucrs/time-series-data'. Please train a forecast model to predict the sales for the next 7 days."
    python agent.py  --no-plan # Use default prompt but skip the planning step (go straight to execution).
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, PlanningStep
from smolagents.utils import AgentError

from tools import (
    create_forecast_plot,
    download_dataset_from_hub,
    forecast_next_7_days,
    generate_final_report,
    preprocess_time_series_data,
    train_xgboost_forecaster,
)

# Load environment variables from .env file
load_dotenv()

# Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL_ID = "deepseek/deepseek-v4-flash"

if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY not found. "
        "Create a .env file with: DEEPSEEK_API_KEY=your-key-here"
    )


# Model (shared by planner and executor, i.e., we use deepseek for both planning and execution)
model = LiteLLMModel(
    model_id=DEEPSEEK_MODEL_ID,
    api_key=DEEPSEEK_API_KEY,
    api_base="https://api.deepseek.com",
)


# Planner agent — creates a plan/TODO list before execution, with user approval
def on_plan_created(memory_step, agent):
    """Step callback: fired after the planner creates a plan.
    Displays the plan and asks the user to approve or cancel."""
    if isinstance(memory_step, PlanningStep):
        plan = memory_step.plan
        print("\n" + "=" * 60)
        print("📋  PLAN")
        print("=" * 60)
        print(plan)
        print("=" * 60)

        # Ask user for approval
        while True:
            choice = input("\nApprove this plan? [Y/n]: ").strip().lower()
            if choice in ("", "y", "yes"):
                print("✅ Plan approved. Executing…\n")
                return  # let the agent continue
            elif choice in ("n", "no"):
                print("❌ Execution cancelled by user.")
                raise AgentError("Plan rejected by user.", agent.logger)
            print("Please enter Y or N.")


planner = CodeAgent(
    tools=[
        download_dataset_from_hub,
        preprocess_time_series_data,
        train_xgboost_forecaster,
        forecast_next_7_days,
        create_forecast_plot,
        generate_final_report,
    ],
    model=model,
    additional_authorized_imports=["datasets", "pandas", "numpy", "pickle", "os", "shutil", "datetime"],
    planning_interval=1000,  # plan on step 1 only; 1000 > max_steps so modulo never fires
    step_callbacks={PlanningStep: on_plan_created},
    stream_outputs=True,  # show tokens as they arrive
    max_steps=15,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A simple DeepSeek-powered agent",
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
        "--no-plan",
        action="store_true",
        help="Skip the planning step entirely (run CodeAgent directly).",
    )
    args = parser.parse_args()

    # Pick the agent to run
    if args.no_plan:
        runner = CodeAgent(
            tools=planner.tools.values(),
            model=model,
            additional_authorized_imports=["datasets", "pandas", "numpy", "pickle", "os", "shutil", "datetime"],
            stream_outputs=True,
            max_steps=20,
        )
        print("⚡ Planner skipped — running CodeAgent directly.\n")
    else:
        runner = planner

    try:
        result = runner.run(args.prompt)
        print(f"\n🎉 Agent result: {result}")
    except Exception as e:
        if "interrupted" in str(e).lower():
            print("\n🛑 Agent was interrupted by user.")
            sys.exit(0)
        raise
