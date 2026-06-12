"""
Utility helpers for the smolagent.

Contents:
    - `load_skill`                  -> Load SKILL.md content from a named skill folder.
    - `load_memory_for_task`        -> Prepend persistent memory content to the task prompt.
    - `EnvConfig`                   -> Dataclass to hold environment configuration loaded from .env.
    - `setup_environment`           -> Load environment variables and return an `EnvConfig` dataclass.
    - `_build_model`                -> Internal function to create a LiteLLMModel based on the selected backend and environment config.
    - `on_plan_created`             -> Step callback to display the plan and ask the user to approve or reject it.
    - `remind_memory_on_plan`       -> Step callback to remind the agent to checkpoint memory after planning.
    - `remind_memory_on_complete`   -> Step callback to nudge the agent to persist learnings after task completion.
    - `make_code_agent`             -> Factory function to create a CodeAgent with the right model and tools based on the selected backend.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from smolagents import CodeAgent, FinalAnswerStep, LiteLLMModel, PlanningStep
from smolagents.utils import AgentError

from tools import MAX_MEMORY_CHARS, MEMORY_FILE, MEMORY_INSTRUCTIONS

# Directory where skill folders live (e.g., skills/forecast/, skills/research/)
SKILLS_DIR = Path(__file__).parent / "skills"


def load_skill(name: str) -> str:
    """Load the SKILL.md content for a named skill.

    Args:
        name: Skill folder name (e.g. 'forecast', 'research').

    Returns:
        The full text content of the SKILL.md file.

    Raises:
        FileNotFoundError: If no folder or SKILL.md exists for that name.
    """
    skill_path = SKILLS_DIR / name / "SKILL.md"
    if not skill_path.exists():
        available = sorted(
            d.name
            for d in SKILLS_DIR.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        )
        listing = ", ".join(available) if available else "(none found)"
        raise FileNotFoundError(
            f"Skill '{name}' not found. "
            f"Available skills: {listing}. "
            f"Looked in: {skill_path}"
        )
    return skill_path.read_text().strip()


def load_memory_for_task(task: str) -> str:
    """Prepend persistent memory content to the task so it's always in context."""
    if not MEMORY_FILE.exists():
        return task
    content = MEMORY_FILE.read_text().strip()
    if not content:
        return task
    if len(content) > MAX_MEMORY_CHARS:
        content = (
            content[: MAX_MEMORY_CHARS // 2]
            + "\n\n... [truncated] ...\n\n"
            + content[-MAX_MEMORY_CHARS // 2 :]
        )
    return (
        f"## Persistent Memory (from MEMORY.md — {len(content)} chars)\n\n"
        f"{content}\n\n"
        f"---\n\n"
        f"## Current Task\n\n"
        f"{task}"
    )


@dataclass
class EnvConfig:
    """Holds all environment-derived configuration."""

    deepseek_api_key: str | None
    deepseek_model_id: str
    openai_api_key: str | None
    openai_model_id: str
    api_base_deepseek: str
    api_base_openai: str


def setup_environment() -> EnvConfig:
    """
    Load environment variables from .env and return an EnvConfig.

    Does NOT validate keys here — validation is deferred to _build_model()
    so that only the selected backend's key is checked.
    """
    load_dotenv()

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not deepseek_api_key and not openai_api_key:
        raise EnvironmentError(
            "No API keys found. Please set at least one of DEEPSEEK_API_KEY "
            "or OPENAI_API_KEY in your .env file."
        )

    return EnvConfig(
        deepseek_api_key=deepseek_api_key,
        deepseek_model_id="deepseek/deepseek-v4-flash",
        openai_api_key=openai_api_key,
        openai_model_id="openai/gpt-5.4-mini-2026-03-17",
        api_base_deepseek="https://api.deepseek.com",
        api_base_openai="http://131.220.150.238:8080",
    )


def _build_model(env: EnvConfig, backend: str) -> LiteLLMModel:
    """Return a LiteLLMModel for the given backend name.

    Raises:
        RuntimeError: If the API key for the selected backend is missing.
    """
    if backend == "deepseek":
        if not env.deepseek_api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY not found. "
                "Create a .env file with: DEEPSEEK_API_KEY=your-key-here"
            )
        return LiteLLMModel(
            model_id=env.deepseek_model_id,
            api_key=env.deepseek_api_key,
            api_base=env.api_base_deepseek,
        )
    elif backend == "openai":
        if not env.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. "
                "Create a .env file with: OPENAI_API_KEY=your-key-here"
            )
        return LiteLLMModel(
            model_id=env.openai_model_id,
            api_key=env.openai_api_key,
            api_base=env.api_base_openai,
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: openai, deepseek")


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

        while True:
            choice = input("\nApprove this plan? [Y/n]: ").strip().lower()
            if choice in ("", "y", "yes"):
                print("✅ Plan approved. Executing…\n")
                return
            elif choice in ("n", "no"):
                print("❌ Execution cancelled by user.")
                raise AgentError("Plan rejected by user.", agent.logger)
            print("Please enter Y or N.")


def remind_memory_on_plan(memory_step, agent):
    """After a planning step, remind the agent to checkpoint its state."""
    if isinstance(memory_step, PlanningStep):
        print(
            "\n🧠  Memory: consider calling `update_memory()` to checkpoint progress.\n"
        )


def remind_memory_on_complete(memory_step, agent):
    """After the final answer, nudge the agent to persist learnings to MEMORY.md."""
    if isinstance(memory_step, FinalAnswerStep):
        print("\n💾  Task complete — persist learnings with `update_memory()`.\n")


def make_code_agent(
    env: EnvConfig,
    backend: str,
    with_planning: bool = True,
    max_steps: int = 15,
    planning_interval: int = 1000,
    tools: list | None = None,
    additional_authorized_imports: list[str] | None = None,
    skill: str | None = None,
    executor_timeout: int | None = 120,
) -> CodeAgent:
    """Create a CodeAgent.

    Args:
        env:               Environment configuration from setup_environment().
        backend:           'openai' or 'deepseek'.
        with_planning:     Whether to include a planning step.
        max_steps:         Maximum steps the agent may take.
        planning_interval: Re-plan every N steps (default 1000 = only on step 1).
        tools:             Custom tool list.
        additional_authorized_imports:  Custom import allowlist.
        skill:             Optional skill name to load from skills/<name>/SKILL.md
                           and inject into the agent's instructions.
        executor_timeout:  Max seconds per tool execution step.  smolagents
                           defaults to 30 s — we raise it to 120 s. Set to None to 
                           disable the timeout entirely.
    """
    model = _build_model(env, backend)

    # Build instructions — start with skill content (if any), then memory instructions
    instructions_parts = []
    if skill:
        skill_content = load_skill(skill)
        instructions_parts.append(f"## Loaded Skill — {skill}\n\n{skill_content}\n")
        line_count = len(skill_content.splitlines())
        print(
            f"📘 Skill '{skill}' loaded "
            f"({len(skill_content):,} chars, {line_count} lines)"
        )
    instructions_parts.append(MEMORY_INSTRUCTIONS)
    instructions = "\n\n---\n\n".join(instructions_parts)

    kwargs: dict = dict(
        tools=tools if tools is not None else [],
        model=model,
        additional_authorized_imports=(
            additional_authorized_imports
            if additional_authorized_imports is not None
            else []
        ),
        stream_outputs=True,  # because it looks cooler!
        max_steps=max_steps,
        instructions=instructions,
        executor_kwargs={"timeout_seconds": executor_timeout},
    )

    if with_planning:
        kwargs["planning_interval"] = planning_interval
        kwargs["step_callbacks"] = {
            PlanningStep: [on_plan_created, remind_memory_on_plan],
            FinalAnswerStep: remind_memory_on_complete,
        }
    else:
        kwargs["step_callbacks"] = {
            FinalAnswerStep: remind_memory_on_complete,
        }

    return CodeAgent(**kwargs)
