# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AI-powered config error helper for trtllm-serve.

When a Pydantic ValidationError occurs due to invalid config, this module
can invoke the Claude CLI to generate a corrective recommendation and
optionally apply the fix interactively.

Activated by setting the environment variable TRTLLM_AI_CONFIG_HELP=1.
"""

import difflib
import json
import os
import re
import shutil
import subprocess  # nosec B404
import textwrap
from typing import Optional, Union

import click
from pydantic import ValidationError

from tensorrt_llm.logger import logger

ConfigError = Union[ValidationError, ValueError, Exception]

_CLAUDE_TIMEOUT_SECONDS = 120

_ENV_VAR = "TRTLLM_AI_CONFIG_HELP"

_YAML_FIX_FENCE = "yaml_fix"


def _is_claude_available() -> bool:
    return shutil.which("claude") is not None


def _format_validation_error(exc: ConfigError) -> str:
    """Format a config error into a concise, readable string."""
    if isinstance(exc, ValidationError):
        lines = []
        for error in exc.errors():
            loc = " -> ".join(str(part) for part in error["loc"])
            lines.append(
                f"  Field: {loc}\n"
                f"    Error: {error['msg']}\n"
                f"    Type: {error['type']}\n"
                f"    Input: {error.get('input', 'N/A')}")
        return "\n".join(lines)
    return f"  {type(exc).__name__}: {exc}"


def _format_validation_error_colored(exc: ConfigError) -> str:
    """Format a config error with click colors."""
    if isinstance(exc, ValidationError):
        lines = []
        for error in exc.errors():
            loc = " -> ".join(str(part) for part in error["loc"])
            field_line = "  Field: " + click.style(loc, fg="yellow")
            error_line = "    Error: " + click.style(error['msg'], fg="red")
            type_line = f"    Type: {error['type']}"
            input_line = f"    Input: {error.get('input', 'N/A')}"
            lines.append(
                f"{field_line}\n{error_line}\n{type_line}\n{input_line}")
        return "\n".join(lines)
    error_type = click.style(type(exc).__name__, fg="yellow")
    error_msg = click.style(str(exc), fg="red")
    return f"  {error_type}: {error_msg}"


def _build_prompt(validation_error: ConfigError,
                  cli_args: dict,
                  config_file_path: Optional[str],
                  config_file_contents: Optional[str],
                  backend: str,
                  extra_instructions: Optional[str] = None) -> str:
    """Build the prompt to send to Claude for config fix recommendations."""
    formatted_error = _format_validation_error(validation_error)

    cli_args_str = " ".join(f"--{k} {v}" for k, v in cli_args.items()
                            if v is not None)

    schema_info = _get_schema_for_backend(backend)

    config_section = ""
    if config_file_path and config_file_contents:
        config_section = textwrap.dedent(f"""\
            ## User's Config File ({config_file_path})
            ```yaml
            {config_file_contents}
            ```
            """)

    extra_section = ""
    if extra_instructions:
        extra_section = textwrap.dedent(f"""\
            ## Additional User Instructions
            {extra_instructions}
            """)

    prompt = textwrap.dedent(f"""\
        You are a TensorRT-LLM configuration expert. A user ran `trtllm-serve`
        and got a Pydantic validation error. Analyze the error and suggest the
        corrected configuration.

        ## Validation Error
        {formatted_error}

        ## User's CLI Command
        trtllm-serve {cli_args_str}

        {config_section}
        ## Available Configuration Fields (JSON Schema)
        ```json
        {schema_info}
        ```

        {extra_section}
        Please provide:
        1. A brief explanation of what went wrong
        2. The corrected config file (YAML) or corrected CLI command
        3. Any additional tips for this configuration

        IMPORTANT: If you provide a corrected YAML config file, you MUST wrap it
        in a fenced code block using ~~~{_YAML_FIX_FENCE} and ~~~ markers, like:

        ~~~{_YAML_FIX_FENCE}
        key: value
        nested:
          field: value
        ~~~

        FORMATTING RULES — your output will be displayed directly in a terminal:
        - Do NOT use any Markdown formatting (no **, no `, no #, no ---, no []()).
        - Use plain text only. Use UPPERCASE or indentation for emphasis.
        - Use simple numbered lists (1. 2. 3.) or dashes (- ) for bullet points.
        - For inline references to config keys or commands, just write them
          plainly without backticks (e.g. "free_gpu_memory_fraction" not
          "`free_gpu_memory_fraction`").
        - Keep your response concise and actionable.
        """)
    return prompt


def _get_schema_for_backend(backend: str) -> str:
    """Get a compact JSON schema for the relevant LlmArgs class."""
    try:
        if backend == "tensorrt":
            from tensorrt_llm.llmapi.llm_args import TrtLlmArgs
            schema = TrtLlmArgs.model_json_schema()
        else:
            from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
            schema = TorchLlmArgs.model_json_schema()

        top_level_fields = {}
        for name, prop in schema.get("properties", {}).items():
            entry = {"type": prop.get("type", prop.get("anyOf", "unknown"))}
            if "description" in prop:
                entry["description"] = prop["description"][:120]
            if "default" in prop:
                entry["default"] = prop["default"]
            top_level_fields[name] = entry

        return json.dumps(top_level_fields, indent=2, default=str)[:8000]
    except Exception:
        return "{schema unavailable}"


def _extract_corrected_yaml(recommendation: str) -> Optional[str]:
    """Extract corrected YAML from Claude's response.

    Looks for content between ~~~yaml_fix and ~~~ fence markers.
    Falls back to ```yaml fenced blocks if the custom markers are absent.
    """
    pattern = rf"~~~{_YAML_FIX_FENCE}\s*\n(.*?)~~~"
    match = re.search(pattern, recommendation, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern_fallback = r"```ya?ml\s*\n(.*?)```"
    match = re.search(pattern_fallback, recommendation, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def _invoke_claude(prompt: str) -> Optional[str]:
    """Invoke the Claude CLI and return its output."""
    try:
        result = subprocess.run(  # nosec B603
            [
                "claude",
                "-p",
                prompt,
                "--output-format",
                "text",
            ],
            capture_output=True,
            text=True,
            timeout=_CLAUDE_TIMEOUT_SECONDS,
        )
        output = result.stdout.strip() if result.stdout else ""
        if result.returncode != 0 or not output:
            if result.stderr and result.stderr.strip():
                logger.warning(
                    f"Claude CLI error: {result.stderr.strip()}")
            return None
        if output.startswith("Error:"):
            logger.warning(f"Claude CLI returned an error: {output}")
            return None
        return output
    except subprocess.TimeoutExpired:
        logger.warning(
            f"Claude CLI timed out after {_CLAUDE_TIMEOUT_SECONDS}s")
        return None
    except Exception as e:
        logger.warning(f"Failed to invoke Claude CLI: {e}")
        return None


def _apply_yaml_fix(corrected_yaml: str,
                    config_file_path: Optional[str]) -> Optional[str]:
    """Write the corrected YAML to the config file, backing up the original.

    Returns the path the corrected YAML was written to, or None on failure.
    """
    if not config_file_path:
        config_file_path = "trtllm_serve_config_fix.yaml"

    backup_path = config_file_path + ".bak"
    try:
        if os.path.exists(config_file_path):
            shutil.copy2(config_file_path, backup_path)
            click.echo(
                click.style(f"  Backed up original config to {backup_path}",
                            fg="green"))

        with open(config_file_path, 'w') as f:
            f.write(corrected_yaml + "\n")
        click.echo(
            click.style(f"  Wrote corrected config to {config_file_path}",
                        fg="green"))
        return config_file_path
    except OSError as e:
        click.echo(
            click.style(f"  Failed to write config file: {e}", fg="red"),
            err=True)
        return None


def _colorize_yaml_diff(original: Optional[str], corrected: str) -> str:
    """Produce a colored inline diff of original vs corrected YAML.

    Added lines are green, removed lines are red with strikethrough,
    unchanged lines are default. If no original is available, all lines
    are shown in green as new content.
    """
    corrected_lines = corrected.splitlines()

    if not original or not original.strip():
        return "\n".join(
            click.style(f"  + {line}", fg="green") for line in corrected_lines)

    original_lines = original.strip().splitlines()
    diff = list(
        difflib.unified_diff(original_lines,
                             corrected_lines,
                             lineterm="",
                             n=9999))

    if len(diff) < 3:
        return "\n".join(
            click.style(f"    {line}", fg="green") for line in corrected_lines)

    output = []
    for line in diff[2:]:
        if line.startswith("+"):
            output.append(click.style(f"  + {line[1:]}", fg="green"))
        elif line.startswith("-"):
            output.append(click.style(f"  - {line[1:]}", fg="red"))
        elif line.startswith(" "):
            output.append(f"    {line[1:]}")
    return "\n".join(output)


def _render_recommendation(recommendation: str,
                           original_config: Optional[str]) -> None:
    """Print the recommendation with the YAML block diff-highlighted."""
    yaml_pattern = rf"~~~{_YAML_FIX_FENCE}\s*\n(.*?)~~~"
    match = re.search(yaml_pattern, recommendation, re.DOTALL)

    if not match:
        click.echo(recommendation, err=True)
        return

    before = recommendation[:match.start()].rstrip()
    after = recommendation[match.end():].lstrip()
    corrected_yaml = match.group(1).strip()

    if before:
        click.echo(before, err=True)
        click.echo(err=True)

    click.echo(click.style("  CORRECTED CONFIG:", fg="cyan", bold=True),
               err=True)
    click.echo(click.style("  " + "-" * 40, fg="cyan"), err=True)
    click.echo(_colorize_yaml_diff(original_config, corrected_yaml), err=True)
    click.echo(click.style("  " + "-" * 40, fg="cyan"), err=True)

    if after:
        click.echo(err=True)
        click.echo(after, err=True)


def _print_separator(color: str = "red") -> None:
    click.echo(click.style("=" * 72, fg=color, bold=True), err=True)


def _prompt_user_action() -> tuple:
    """Prompt the user for an action.

    Returns:
        ("a", None) to apply, ("q", None) to quit, or
        ("r", instructions) to revise with the user's input as instructions.
    """
    click.echo(err=True)
    click.echo(click.style("What would you like to do?", bold=True), err=True)
    click.echo("  " + click.style("[A]", fg="cyan", bold=True) +
               "pply fix and restart",
               err=True)
    click.echo("  " + click.style("[Q]", fg="cyan", bold=True) + "uit",
               err=True)
    click.echo("  Or type your instructions to revise the suggestion.",
               err=True)
    click.echo(err=True)

    choice = click.prompt(
        click.style("A / Q / revision instructions", fg="cyan", bold=True),
        type=str,
        default="q",
        err=True).strip()

    lower = choice.lower()
    if lower in ("a", "apply"):
        return ("a", None)
    if lower in ("q", "quit"):
        return ("q", None)
    return ("r", choice)


def handle_validation_error(exc: ConfigError,
                            cli_args: dict,
                            config_file_path: Optional[str] = None,
                            config_file_contents: Optional[str] = None,
                            backend: str = "pytorch") -> bool:
    """Handle a config validation error with optional AI-powered suggestions.

    Always prints the formatted error. When TRTLLM_AI_CONFIG_HELP=1 is set
    and the Claude CLI is available, invokes Claude to generate a corrective
    recommendation and offers an interactive fix-and-restart loop.

    Args:
        exc: The error that was raised (ValidationError, ValueError, etc.).
        cli_args: Dict of CLI arguments the user passed to trtllm-serve.
        config_file_path: Path to the YAML config file, if one was provided.
        config_file_contents: Raw contents of the config file, if available.
        backend: The backend string ("pytorch", "tensorrt", "_autodeploy").

    Returns:
        True if the user chose to apply a fix and wants to retry, False
        otherwise.
    """
    click.echo(err=True)
    _print_separator("red")
    click.echo(click.style("  CONFIG VALIDATION ERROR", fg="red", bold=True),
               err=True)
    _print_separator("red")
    click.echo(err=True)
    click.echo(_format_validation_error_colored(exc), err=True)
    click.echo(err=True)
    _print_separator("red")
    click.echo(err=True)

    ai_help_enabled = os.environ.get(_ENV_VAR, "").strip() in ("1", "true",
                                                                "yes")

    if not ai_help_enabled:
        click.echo(
            click.style(
                f"Hint: Set {_ENV_VAR}=1 to get AI-powered config suggestions "
                "via Claude CLI.",
                dim=True),
            err=True)
        return False

    if not _is_claude_available():
        click.echo(click.style(
            "AI config help is enabled but the 'claude' CLI was not found "
            "on PATH.",
            fg="yellow"),
                   err=True)
        click.echo(click.style(
            "Install it from: https://claude.ai/install.sh", dim=True),
                   err=True)
        return False

    extra_instructions: Optional[str] = None

    while True:
        click.echo(click.style(
            "Generating AI-powered config recommendation...", fg="cyan"),
                   err=True)
        click.echo(err=True)

        prompt = _build_prompt(exc, cli_args, config_file_path,
                               config_file_contents, backend,
                               extra_instructions)

        recommendation = _invoke_claude(prompt)

        if not recommendation:
            click.echo(click.style(
                "Could not generate an AI recommendation. "
                "Please review the validation error above.",
                fg="yellow"),
                       err=True)
            return False

        _print_separator("green")
        click.echo(click.style("  AI CONFIG RECOMMENDATION (via Claude)",
                               fg="green",
                               bold=True),
                   err=True)
        _print_separator("green")
        click.echo(err=True)
        _render_recommendation(recommendation, config_file_contents)
        click.echo(err=True)
        _print_separator("green")

        action, user_input = _prompt_user_action()

        if action == "q":
            return False

        if action == "r":
            extra_instructions = user_input
            click.echo(err=True)
            continue

        if action == "a":
            corrected_yaml = _extract_corrected_yaml(recommendation)
            if not corrected_yaml:
                click.echo(click.style(
                    "Could not extract corrected YAML from the "
                    "recommendation. Please apply the fix manually.",
                    fg="yellow"),
                           err=True)
                return False

            click.echo(err=True)
            click.echo(click.style("Applying fix...", fg="green", bold=True),
                       err=True)
            written_path = _apply_yaml_fix(corrected_yaml, config_file_path)
            if not written_path:
                return False

            click.echo(
                click.style("  Restarting trtllm-serve...",
                            fg="green",
                            bold=True),
                err=True)
            click.echo(err=True)
            return True
