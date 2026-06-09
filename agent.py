"""
A smolagents agent with pluggable backends (DeepSeek or Mistral).

Usage:
    python agent.py  # Uses Mistral by default with the built-in prompt
    python agent.py --backend deepseek  # Use DeepSeek instead
    python agent.py --no-plan  # Skip the planning step (go straight to execution)
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
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL_ID = "mistral/mistral-small-latest"

API_BASE_DEEPSEEK = "https://api.deepseek.com"
API_BASE_MISTRAL = "http://131.220.150.238:8080"

if not DEEPSEEK_API_KEY:
    print(
        "Warning: DEEPSEEK_API_KEY not found. "
        "DeepSeek backend will not be available. "
        "Create a .env file with: DEEPSEEK_API_KEY=your-key-here"
    )

if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY not found. "
        "Create a .env file with: MISTRAL_API_KEY=your-key-here"
    )


# Backend registry – maps CLI names to LiteLLMModel constructors
def _build_model(backend: str) -> LiteLLMModel:
    """Return a LiteLLMModel for the given backend name."""
    backends = {
        "deepseek": lambda: LiteLLMModel(
            model_id=DEEPSEEK_MODEL_ID,
            api_key=DEEPSEEK_API_KEY,
            api_base=API_BASE_DEEPSEEK,
        ),
        "mistral": lambda: LiteLLMModel(
            model_id=MISTRAL_MODEL_ID,
            api_key=MISTRAL_API_KEY,
            api_base=API_BASE_MISTRAL,
        ),
    }
    if backend not in backends:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: {', '.join(backends)}"
        )
    return backends[backend]()


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


TOOLS = [
    download_dataset_from_hub,
    preprocess_time_series_data,
    train_xgboost_forecaster,
    forecast_next_7_days,
    create_forecast_plot,
    generate_final_report,
]

AUTHORIZED_IMPORTS = [
    "datasets", "pandas", "numpy", "pickle", "os", "shutil", "datetime",
]


def _make_code_agent(model: LiteLLMModel, with_planning: bool = True, max_steps: int = 15) -> CodeAgent:
    """Create a CodeAgent, optionally with planning."""
    kwargs: dict = dict(
        tools=TOOLS,
        model=model,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        stream_outputs=True,
        max_steps=max_steps,
    )
    if with_planning:
        kwargs["planning_interval"] = 1000  # plan on step 1 only
        kwargs["step_callbacks"] = {PlanningStep: on_plan_created}
    return CodeAgent(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A smolagents agent implementation.",
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
        choices=["mistral", "deepseek"],
        default="mistral",
        help="LLM backend to use (default: %(default)s).",
    )
    parser.add_argument(
        "--no-plan",
        action="store_true",
        help="Skip the planning step entirely (run CodeAgent directly).",
    )
    args = parser.parse_args()

    # Build the model for the chosen backend
    model = _build_model(args.backend)
    print(f"🤖 Using backend: {args.backend}\n")

    # Pick the agent to run
    if args.no_plan:
        runner = _make_code_agent(model, with_planning=False, max_steps=20)
        print("⚡ Planner skipped — running CodeAgent directly.\n")
    else:
        runner = _make_code_agent(model, with_planning=True, max_steps=15)

    try:
        result = runner.run(args.prompt)
        print(f"\n🎉 Agent result: {result}")
    except Exception as e:
        if "interrupted" in str(e).lower():
            print("\n🛑 Agent was interrupted by user.")
            sys.exit(0)
        raise
