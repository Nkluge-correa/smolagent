"""
Utility helpers for the smolagent.

Contents:
    - `load_skill`                  -> Load SKILL.md content from a named skill folder.
    - `load_memory_for_task`        -> Prepend persistent memory content to the task prompt.
    - `load_system_prompt`          -> Load prompt templates from SYSTEM.yaml (or a custom path).
    - `EnvConfig`                   -> Dataclass to hold environment configuration loaded from .env.
    - `setup_environment`           -> Load environment variables and return an `EnvConfig` dataclass.
    - `_build_model`                -> Internal function to create a LiteLLMModel based on the selected backend and environment config.
    - `approve_plan`                -> Display a plan and ask the user to approve or reject it.
    - `remind_memory_on_complete`   -> Step callback to nudge the agent to persist learnings after task completion.
    - `save_session_trace`          -> Save the full agentic trace as a JSON file in the traces/ directory.
    - `make_planner_agent`          -> Factory for a lightweight planning agent (no tools, just reasoning).
    - `make_code_agent`             -> Factory for the executor agent (with tools, no built-in planning).
"""

import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from smolagents import CodeAgent, FinalAnswerStep, LiteLLMModel

from tools import MAX_MEMORY_CHARS, MEMORY_FILE, MEMORY_INSTRUCTIONS

# Directory where skill folders live (e.g., skills/forecast/, skills/research/)
SKILLS_DIR = Path(__file__).parent / "skills"

# Directory where session traces are saved as JSON files
TRACES_DIR = Path(__file__).parent / "traces"

# Default system prompt file: loaded automatically by make_code_agent().
# Edit this file to customize the agent's system prompt while keeping all
# Jinja2 placeholders ({{tools}}, {{authorized_imports}}, etc.) intact.
EXECUTOR_FILE = Path(__file__).parent / "prompts" / "EXECUTOR.yaml"

# Planner-specific prompt: a simplified system prompt for the planner agent
# (which has NO tools).  Loaded automatically by make_planner_agent().
PLANNER_FILE = Path(__file__).parent / "prompts" / "PLANNER.yaml"


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
            d.name for d in SKILLS_DIR.iterdir() if d.is_dir() and (d / "SKILL.md").exists()
        )
        listing = ", ".join(available) if available else "(none found)"
        raise FileNotFoundError(
            f"Skill '{name}' not found. Available skills: {listing}. Looked in: {skill_path}"
        )
    return skill_path.read_text().strip()


def load_system_prompt(path: Path | str | None = None) -> dict[str, Any]:
    """Load prompt templates from a YAML file.

    By default, loads from `prompts/EXECUTOR.yaml` in the repo root.  Pass a custom
    path to load a different prompt template file.

    The returned dict has the same shape as smolagents' PromptTemplates
    TypedDict (system_prompt, planning, managed_agent, final_answer) and
    can be passed directly to CodeAgent(..., prompt_templates=...).

    Args:
        path: Path to a YAML file containing prompt templates.
              If None, uses the default `prompts/EXECUTOR.yaml`.

    Returns:
        A dict of prompt templates ready to pass to CodeAgent.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    file_path = Path(path) if path else EXECUTOR_FILE
    if not file_path.exists():
        raise FileNotFoundError(
            f"System prompt file not found: {file_path}\n"
            f"Create a SYSTEM.yaml file or pass a custom path to load_system_prompt()."
        )
    with open(file_path, encoding="utf-8") as f:
        templates = yaml.safe_load(f)
    print(f"📋 System prompt loaded from: {file_path}")
    return templates


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
        raise OSError(
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
                "OPENAI_API_KEY not found. Create a .env file with: OPENAI_API_KEY=your-key-here"
            )
        return LiteLLMModel(
            model_id=env.openai_model_id,
            api_key=env.openai_api_key,
            api_base=env.api_base_openai,
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: openai, deepseek")


def approve_plan(plan: str) -> str:
    """Display a plan and ask the user to approve or reject it.

    Args:
        plan: The plan text generated by the planner agent.

    Returns:
        The plan text if approved.

    Raises:
        SystemExit: If the user rejects the plan.
    """
    print("\n" + "=" * 60)
    print("📋  PLAN  (generated by Planner agent)")
    print("=" * 60)
    print(plan)
    print("=" * 60)

    while True:
        choice = input("\nApprove this plan? [Y/n]: ").strip().lower()
        if choice in ("", "y", "yes"):
            print("✅ Plan approved. Handing off to Executor…\n")
            return plan
        elif choice in ("n", "no"):
            print("❌ Plan rejected by user. Exiting.")
            sys.exit(0)
        print("Please enter Y or N.")


def remind_memory_on_complete(memory_step, agent):
    """After the final answer, nudge the agent to persist learnings to MEMORY.md."""
    if isinstance(memory_step, FinalAnswerStep):
        print("\n💾  Task complete — persist learnings with `update_memory()`.\n")


def save_session_trace(
    agent: CodeAgent,
    task: str,
    backend: str,
    state: str = "success",
    output: Any = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save the full agentic trace from an agent run as a timestamped JSON file.

    The trace includes:
      - Metadata (session ID, timestamp, backend, agent config)
      - The original task
      - Final output and run state
      - Full step-by-step trace from agent memory (model inputs/outputs,
        tool calls, observations, errors)
      - Token usage statistics per step and total
      - Timing information (start, end, duration)

    Args:
        agent:     The CodeAgent instance after a run (or partial run).
        task:      The original task string.
        backend:   The LLM backend used ('openai' or 'deepseek').
        state:     Final state: 'success', 'max_steps_error', or 'interrupted'.
        output:    Final output from the agent (if available).
        extra_metadata: Optional extra key-value pairs to include.

    Returns:
        The Path to the saved trace file.
    """
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    session_id = uuid.uuid4().hex[:12]
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y-%m-%dT%H%M%SZ")
    filename = f"trace-{timestamp}-{session_id}.json"
    filepath = TRACES_DIR / filename

    # Gather token usage totals from memory steps
    total_input_tokens = 0
    total_output_tokens = 0
    per_step_tokens: list[dict[str, Any]] = []

    for step in agent.memory.steps:
        tu = getattr(step, "token_usage", None)
        if tu is not None:
            total_input_tokens += getattr(tu, "input_tokens", 0) or 0
            total_output_tokens += getattr(tu, "output_tokens", 0) or 0
            per_step_tokens.append(
                {
                    "step": getattr(step, "step_number", "?"),
                    "input_tokens": getattr(tu, "input_tokens", 0),
                    "output_tokens": getattr(tu, "output_tokens", 0),
                }
            )

    # Gather step dicts: use `get_succinct_steps` steps if you want to avoid huge model_input_messages
    # unless the trace is small enough; full_steps can be very large.
    try:
        steps_data = agent.memory.get_full_steps()
    except Exception:
        steps_data = []

    trace: dict[str, Any] = {
        "session_id": session_id,
        "timestamp": now.isoformat(),
        "backend": backend,
        "model_id": (
            agent.model.model_id if hasattr(agent.model, "model_id") else str(agent.model)
        ),
        "agent_type": type(agent).__name__,
        "max_steps": agent.max_steps,
        "planning_interval": getattr(agent, "planning_interval", None),
        "task": task,
        "state": state,
        "output": str(output) if output is not None else None,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "per_step_token_usage": per_step_tokens,
        "num_steps": len(agent.memory.steps),
        "steps": steps_data,
    }

    if extra_metadata:
        trace["metadata"] = extra_metadata

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n📝 Session trace saved -> {filepath} ({filepath.stat().st_size:,} bytes)")
    return filepath


def make_planner_agent(
    env: EnvConfig,
    backend: str,
    max_steps: int = 5,
    skill: str | None = None,
    prompt_templates: dict[str, Any] | None = None,
    instructions: str | None = None,
) -> CodeAgent:
    """Create a lightweight Planner agent: a separate CodeAgent instance
    whose only job is to reason about the task and produce a step-by-step plan.

    The planner has no tools: it cannot execute anything.  It just thinks
    and writes a plan.  The plan is then handed to a separate executor agent
    created by :func:`make_code_agent`.

    Args:
        env:               Environment configuration from setup_environment().
        backend:           'openai' or 'deepseek'.
        max_steps:         Maximum reasoning steps for the planner (default 5).
        skill:             Optional skill name to load and inject into the
                           planner's instructions.
        prompt_templates:  Optional custom prompt templates.  If None, loaded
                           from PLANNER.yaml (a no-tools planning prompt).
        instructions:      Optional extra instructions appended to the planner's
                           system prompt.

    Returns:
        A CodeAgent configured for planning only (no tools).
    """
    model = _build_model(env, backend)

    if prompt_templates is None:
        prompt_templates = load_system_prompt(PLANNER_FILE)

    # Build planner-specific instructions (skill + any extra instructions).
    # The planner's system prompt (PLANNER.yaml) already teaches the correct
    # output pattern — we don't need to repeat it here.
    parts: list[str] = []
    if skill:
        skill_content = load_skill(skill)
        parts.append(f"## Loaded Skill — {skill}\n\n{skill_content}\n")
        line_count = len(skill_content.splitlines())
        print(f"📘 Skill '{skill}' loaded ({len(skill_content):,} chars, {line_count} lines)")
    if instructions:
        parts.append(instructions)
    planner_instructions = "\n\n---\n\n".join(parts) if parts else None

    print(f"🧠 Planner agent  |  backend: {backend}  |  max plan steps: {max_steps}")

    return CodeAgent(
        tools=[],  # No tools, planning is pure reasoning
        model=model,
        prompt_templates=prompt_templates,
        max_steps=max_steps,
        instructions=planner_instructions,
        stream_outputs=False,
    )


def make_code_agent(
    env: EnvConfig,
    backend: str,
    max_steps: int = 15,
    tools: list | None = None,
    additional_authorized_imports: list[str] | None = None,
    skill: str | None = None,
    executor_timeout: int | None = 120,
    prompt_templates: dict[str, Any] | None = None,
) -> CodeAgent:
    """Create an Executor CodeAgent: the agent that actually runs tools.

    Unlike the planner, this agent is equipped with the full tool set and
    can write/execute Python code.  It receives a plan (generated by a
    separate planner agent) embedded in its task description.

    No built-in planning step: planning is done by a separate
    :func:`make_planner_agent` instance before this agent runs.

    Args:
        env:               Environment configuration from setup_environment().
        backend:           'openai' or 'deepseek'.
        max_steps:         Maximum steps the agent may take.
        tools:             Custom tool list.
        additional_authorized_imports:  Custom import allowlist.
        skill:             Optional skill name to load and inject.
        executor_timeout:  Max seconds per tool execution step.  Set to None
                           to disable the timeout entirely.
        prompt_templates:  Optional custom prompt templates.  If None, loaded
                           from SYSTEM.yaml.

    Returns:
        A CodeAgent configured for execution (with tools, no built-in planning).
    """
    model = _build_model(env, backend)

    if prompt_templates is None:
        prompt_templates = load_system_prompt(EXECUTOR_FILE)

    # Build instructions: skill content (if any), then memory instructions
    instructions_parts = []
    if skill:
        skill_content = load_skill(skill)
        instructions_parts.append(f"## Loaded Skill — {skill}\n\n{skill_content}\n")
        line_count = len(skill_content.splitlines())
        print(f"📘 Skill '{skill}' loaded ({len(skill_content):,} chars, {line_count} lines)")
    instructions_parts.append(MEMORY_INSTRUCTIONS)
    instructions = "\n\n---\n\n".join(instructions_parts)

    print(f"🤖 Executor agent  |  backend: {backend}  |  max steps: {max_steps}")

    return CodeAgent(
        tools=tools if tools is not None else [],
        model=model,
        prompt_templates=prompt_templates,
        additional_authorized_imports=(
            additional_authorized_imports if additional_authorized_imports is not None else []
        ),
        stream_outputs=False,
        max_steps=max_steps,
        instructions=instructions,
        executor_kwargs={"timeout_seconds": executor_timeout},
        step_callbacks={
            FinalAnswerStep: remind_memory_on_complete,
        },
    )
