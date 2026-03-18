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
import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

from pydantic import BaseModel, ValidationError

from tensorrt_llm.commands.ai_config_helper import (
    _apply_yaml_fix,
    _build_prompt,
    _colorize_yaml_diff,
    _extract_corrected_yaml,
    _format_validation_error,
    _invoke_claude,
    _is_claude_available,
    handle_validation_error,
)


class _DummyConfig(BaseModel):
    class Config:
        extra = "forbid"

    batch_size: int = 32
    model_name: str = "gpt2"


def _make_validation_error(**kwargs) -> ValidationError:
    """Create a real Pydantic ValidationError by passing invalid data."""
    try:
        _DummyConfig(**kwargs)
    except ValidationError as exc:
        return exc
    raise AssertionError("Expected ValidationError was not raised")


class TestFormatValidationError:

    def test_extra_field(self):
        exc = _make_validation_error(nonexistent_field="bad")
        result = _format_validation_error(exc)
        assert "nonexistent_field" in result
        assert "Extra inputs are not permitted" in result

    def test_wrong_type(self):
        exc = _make_validation_error(batch_size="not_an_int")
        result = _format_validation_error(exc)
        assert "batch_size" in result
        assert "int" in result.lower()

    def test_plain_value_error(self):
        exc = ValueError("LLM got invalid argument: enable_block_reuse")
        result = _format_validation_error(exc)
        assert "enable_block_reuse" in result
        assert "ValueError" in result


class TestBuildPrompt:

    def test_contains_error_and_cli_args(self):
        exc = _make_validation_error(nonexistent_field="bad")
        prompt = _build_prompt(
            exc,
            cli_args={"model": "meta-llama/Llama-2-7b", "backend": "pytorch"},
            config_file_path=None,
            config_file_contents=None,
            backend="pytorch",
        )
        assert "nonexistent_field" in prompt
        assert "meta-llama/Llama-2-7b" in prompt
        assert "trtllm-serve" in prompt

    def test_includes_config_file_contents(self):
        exc = _make_validation_error(batch_size="abc")
        config_yaml = "batch_size: abc\nmodel_name: gpt2\n"
        prompt = _build_prompt(
            exc,
            cli_args={"model": "gpt2"},
            config_file_path="/tmp/config.yaml",
            config_file_contents=config_yaml,
            backend="pytorch",
        )
        assert "config.yaml" in prompt
        assert "batch_size: abc" in prompt

    def test_schema_section_present(self):
        exc = _make_validation_error(nonexistent_field="bad")
        prompt = _build_prompt(
            exc,
            cli_args={"model": "gpt2"},
            config_file_path=None,
            config_file_contents=None,
            backend="pytorch",
        )
        assert "Configuration Fields" in prompt

    def test_yaml_fix_fence_instruction_present(self):
        exc = _make_validation_error(nonexistent_field="bad")
        prompt = _build_prompt(
            exc,
            cli_args={"model": "gpt2"},
            config_file_path=None,
            config_file_contents=None,
            backend="pytorch",
        )
        assert "~~~yaml_fix" in prompt

    def test_extra_instructions_included(self):
        exc = _make_validation_error(nonexistent_field="bad")
        prompt = _build_prompt(
            exc,
            cli_args={"model": "gpt2"},
            config_file_path=None,
            config_file_contents=None,
            backend="pytorch",
            extra_instructions="Use fp8 quantization instead.",
        )
        assert "Use fp8 quantization instead." in prompt
        assert "Additional User Instructions" in prompt


class TestExtractCorrectedYaml:

    def test_extracts_yaml_fix_fence(self):
        text = ("Here is the fix:\n\n"
                "~~~yaml_fix\n"
                "batch_size: 32\n"
                "model_name: gpt2\n"
                "~~~\n\n"
                "This should work.")
        result = _extract_corrected_yaml(text)
        assert result == "batch_size: 32\nmodel_name: gpt2"

    def test_extracts_yaml_fix_fence_with_extra_whitespace(self):
        text = ("~~~yaml_fix\n"
                "\n"
                "  batch_size: 64\n"
                "\n"
                "~~~")
        result = _extract_corrected_yaml(text)
        assert result == "batch_size: 64"

    def test_fallback_to_yaml_code_block(self):
        text = ("Here is the fix:\n\n"
                "```yaml\n"
                "batch_size: 32\n"
                "```\n")
        result = _extract_corrected_yaml(text)
        assert result == "batch_size: 32"

    def test_fallback_to_yml_code_block(self):
        text = ("```yml\n"
                "batch_size: 16\n"
                "```\n")
        result = _extract_corrected_yaml(text)
        assert result == "batch_size: 16"

    def test_returns_none_when_no_yaml(self):
        text = "Just set batch_size to 32 in your config."
        result = _extract_corrected_yaml(text)
        assert result is None

    def test_prefers_yaml_fix_over_yaml_block(self):
        text = ("```yaml\n"
                "wrong: value\n"
                "```\n\n"
                "~~~yaml_fix\n"
                "correct: value\n"
                "~~~\n")
        result = _extract_corrected_yaml(text)
        assert result == "correct: value"


class TestColorizeYamlDiff:

    def test_added_lines_are_green(self):
        original = "key1: value1"
        corrected = "key1: value1\nkey2: value2"
        result = _colorize_yaml_diff(original, corrected)
        assert "key2: value2" in result
        assert "+ " in result

    def test_removed_lines_are_red(self):
        original = "key1: value1\nkey2: bad"
        corrected = "key1: value1"
        result = _colorize_yaml_diff(original, corrected)
        assert "- " in result

    def test_no_original_shows_all_green(self):
        result = _colorize_yaml_diff(None, "key: value")
        assert "+ " in result
        assert "key: value" in result

    def test_empty_original_shows_all_green(self):
        result = _colorize_yaml_diff("", "key: value")
        assert "+ " in result

    def test_identical_shows_no_diff_markers(self):
        content = "key: value"
        result = _colorize_yaml_diff(content, content)
        assert "+ " not in result
        assert "- " not in result
        assert "key: value" in result


class TestIsClaudeAvailable:

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    def test_available(self, mock_which):
        assert _is_claude_available() is True

    @patch("shutil.which", return_value=None)
    def test_not_available(self, mock_which):
        assert _is_claude_available() is False


class TestInvokeClaude:

    @patch("subprocess.run")
    def test_successful_invocation(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Use batch_size: 32 instead of 'abc'",
            stderr="",
        )
        result = _invoke_claude("test prompt")
        assert result == "Use batch_size: 32 instead of 'abc'"
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "--output-format" in cmd
        assert "-p" in cmd

    @patch("subprocess.run")
    def test_failed_invocation(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="API key not set",
        )
        result = _invoke_claude("test prompt")
        assert result is None

    @patch("subprocess.run")
    def test_error_prefix_in_stdout(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Error: Reached max turns (1)",
            stderr="",
        )
        result = _invoke_claude("test prompt")
        assert result is None

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired("claude",
                                                                   120))
    def test_timeout(self, mock_run):
        result = _invoke_claude("test prompt")
        assert result is None

    @patch("subprocess.run", side_effect=FileNotFoundError("claude not found"))
    def test_not_installed(self, mock_run):
        result = _invoke_claude("test prompt")
        assert result is None


class TestApplyYamlFix:

    def test_writes_corrected_yaml(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                         delete=False) as f:
            f.write("old: value\n")
            path = f.name
        try:
            result = _apply_yaml_fix("new: value", path)
            assert result == path
            with open(path) as f:
                assert "new: value" in f.read()
            assert os.path.exists(path + ".bak")
            with open(path + ".bak") as f:
                assert "old: value" in f.read()
        finally:
            os.unlink(path)
            if os.path.exists(path + ".bak"):
                os.unlink(path + ".bak")

    def test_creates_new_file_when_no_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = _apply_yaml_fix("key: value", None)
                assert result is not None
                assert os.path.exists(result)
                with open(result) as f:
                    assert "key: value" in f.read()
            finally:
                os.chdir(original_cwd)


class TestHandleValidationError:

    @patch.dict("os.environ", {}, clear=False)
    def test_returns_false_when_env_not_set(self):
        os.environ.pop("TRTLLM_AI_CONFIG_HELP", None)
        exc = _make_validation_error(nonexistent_field="bad")
        result = handle_validation_error(exc, cli_args={"model": "gpt2"})
        assert result is False

    @patch.dict("os.environ", {"TRTLLM_AI_CONFIG_HELP": "1"})
    @patch("tensorrt_llm.commands.ai_config_helper._is_claude_available",
           return_value=False)
    def test_returns_false_when_claude_missing(self, mock_avail):
        exc = _make_validation_error(nonexistent_field="bad")
        result = handle_validation_error(exc, cli_args={"model": "gpt2"})
        assert result is False

    @patch.dict("os.environ", {"TRTLLM_AI_CONFIG_HELP": "1"})
    @patch("tensorrt_llm.commands.ai_config_helper._is_claude_available",
           return_value=True)
    @patch("tensorrt_llm.commands.ai_config_helper._invoke_claude",
           return_value=None)
    def test_returns_false_when_claude_fails(self, mock_invoke, mock_avail):
        exc = _make_validation_error(nonexistent_field="bad")
        result = handle_validation_error(exc, cli_args={"model": "gpt2"})
        assert result is False

    @patch.dict("os.environ", {"TRTLLM_AI_CONFIG_HELP": "1"})
    @patch("tensorrt_llm.commands.ai_config_helper._is_claude_available",
           return_value=True)
    @patch("tensorrt_llm.commands.ai_config_helper._invoke_claude",
           return_value="Set batch_size to 32.\n~~~yaml_fix\nbatch_size: 32\n~~~")
    @patch("tensorrt_llm.commands.ai_config_helper._prompt_user_action",
           return_value=("q", None))
    def test_returns_false_on_quit(self, mock_prompt, mock_invoke, mock_avail):
        exc = _make_validation_error(nonexistent_field="bad")
        result = handle_validation_error(exc, cli_args={"model": "gpt2"})
        assert result is False

    @patch.dict("os.environ", {"TRTLLM_AI_CONFIG_HELP": "1"})
    @patch("tensorrt_llm.commands.ai_config_helper._is_claude_available",
           return_value=True)
    @patch("tensorrt_llm.commands.ai_config_helper._invoke_claude",
           return_value="Fix:\n~~~yaml_fix\nbatch_size: 32\n~~~")
    @patch("tensorrt_llm.commands.ai_config_helper._prompt_user_action",
           return_value=("a", None))
    def test_returns_true_on_apply(self, mock_prompt, mock_invoke, mock_avail):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                         delete=False) as f:
            f.write("batch_size: bad\n")
            path = f.name
        try:
            exc = _make_validation_error(nonexistent_field="bad")
            result = handle_validation_error(
                exc,
                cli_args={"model": "gpt2"},
                config_file_path=path,
                config_file_contents="batch_size: bad\n",
            )
            assert result is True
            with open(path) as f:
                assert "batch_size: 32" in f.read()
        finally:
            os.unlink(path)
            if os.path.exists(path + ".bak"):
                os.unlink(path + ".bak")

    @patch.dict("os.environ", {"TRTLLM_AI_CONFIG_HELP": "1"})
    @patch("tensorrt_llm.commands.ai_config_helper._is_claude_available",
           return_value=True)
    @patch("tensorrt_llm.commands.ai_config_helper._invoke_claude")
    @patch("tensorrt_llm.commands.ai_config_helper._prompt_user_action",
           side_effect=[("r", "Try using enable_block_reuse instead"),
                        ("q", None)])
    def test_revise_then_quit(self, mock_action, mock_invoke, mock_avail):
        mock_invoke.side_effect = [
            "First suggestion.\n~~~yaml_fix\nold: value\n~~~",
            "Revised suggestion.\n~~~yaml_fix\nnew: value\n~~~",
        ]
        exc = _make_validation_error(nonexistent_field="bad")
        result = handle_validation_error(exc, cli_args={"model": "gpt2"})
        assert result is False
        assert mock_invoke.call_count == 2

    @patch.dict("os.environ", {"TRTLLM_AI_CONFIG_HELP": "1"})
    @patch("tensorrt_llm.commands.ai_config_helper._is_claude_available",
           return_value=True)
    @patch("tensorrt_llm.commands.ai_config_helper._invoke_claude",
           return_value="Just set batch_size to 32.")
    @patch("tensorrt_llm.commands.ai_config_helper._prompt_user_action",
           return_value=("a", None))
    def test_apply_fails_when_no_yaml_extracted(self, mock_prompt, mock_invoke,
                                                mock_avail):
        exc = _make_validation_error(nonexistent_field="bad")
        result = handle_validation_error(exc, cli_args={"model": "gpt2"})
        assert result is False

    @patch.dict("os.environ", {}, clear=False)
    def test_handles_value_error(self):
        os.environ.pop("TRTLLM_AI_CONFIG_HELP", None)
        exc = ValueError("LLM got invalid argument: enable_block_reuse")
        result = handle_validation_error(exc, cli_args={"model": "gpt2"})
        assert result is False

    @patch.dict("os.environ", {"TRTLLM_AI_CONFIG_HELP": "1"})
    @patch("tensorrt_llm.commands.ai_config_helper._is_claude_available",
           return_value=True)
    @patch("tensorrt_llm.commands.ai_config_helper._invoke_claude",
           return_value="Remove enable_block_reuse.\n"
                        "~~~yaml_fix\nkv_cache_config:\n"
                        "  free_gpu_memory_fraction: 0.8\n~~~")
    @patch("tensorrt_llm.commands.ai_config_helper._prompt_user_action",
           return_value=("a", None))
    def test_value_error_apply_fix(self, mock_prompt, mock_invoke, mock_avail):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                         delete=False) as f:
            f.write("enable_block_reuse: false\n")
            path = f.name
        try:
            exc = ValueError("LLM got invalid argument: enable_block_reuse")
            result = handle_validation_error(
                exc,
                cli_args={"model": "gpt2"},
                config_file_path=path,
                config_file_contents="enable_block_reuse: false\n",
            )
            assert result is True
            with open(path) as f:
                content = f.read()
            assert "enable_block_reuse" not in content
            assert "free_gpu_memory_fraction" in content
        finally:
            os.unlink(path)
            if os.path.exists(path + ".bak"):
                os.unlink(path + ".bak")
