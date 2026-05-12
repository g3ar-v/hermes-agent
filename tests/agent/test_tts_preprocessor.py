"""Unit tests for agent/tts_preprocessor.py."""

import pytest
from unittest.mock import patch, MagicMock

from agent.tts_preprocessor import preprocess_for_tts, _build_heuristic_summary


class TestBuildHeuristicSummary:
    "Heuristic extraction tests."

    def test_file_creation(self):
        text = "I created file /etc/myapp/config.yaml with the following settings..."
        result = _build_heuristic_summary(text)
        assert "Created" in result

    def test_installed_packages(self):
        text = "Installed 3 packages: flask, gunicorn, redis"
        result = _build_heuristic_summary(text)
        assert "Installed 3 packages" in result

    def test_service_started(self):
        text = "I started the nginx service and it is running on port 80."
        result = _build_heuristic_summary(text)
        assert "Started the nginx" in result

    def test_ran_tests(self):
        text = "Ran 42 tests successfully, all passed."
        result = _build_heuristic_summary(text)
        assert "Ran 42 tests" in result

    def test_multiple_actions(self):
        text = (
            "I created file src/main.py. I installed 2 packages. "
            "I started the api service."
        )
        result = _build_heuristic_summary(text)
        assert "Created" in result
        assert "Installed" in result
        assert "Started" in result

    def test_fallback_first_sentence(self):
        text = "Here is the configuration for the application. ```python\n..."
        result = _build_heuristic_summary(text)
        assert "Here is the configuration" in result

    def test_empty_text(self):
        assert _build_heuristic_summary("") == ""

    def test_max_length(self):
        text = "A" * 2000
        result = _build_heuristic_summary(text, max_len=100)
        assert len(result) <= 100


class TestPreprocessForTts:
    "Main entry point tests."

    def test_mode_off_passthrough(self):
        text = "Hello world"
        assert preprocess_for_tts(text, mode="off") == text

    def test_mode_heuristic(self):
        text = "Installed 5 packages and started the nginx service."
        result = preprocess_for_tts(text, mode="heuristic")
        assert "Installed" in result or "nginx" in result

    def test_mode_llm_uses_call_llm(self):
        """LLM mode calls call_llm and returns the summary."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Set up the project."

        with patch("agent.auxiliary_client.call_llm", return_value=mock_response):
            result = preprocess_for_tts(
                "I installed all dependencies and created the config.",
                mode="llm",
            )
            assert "Set up the project" in result

    def test_mode_llm_fallback_on_failure(self):
        """LLM failure falls back to heuristic, no exception."""
        with patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("fail")):
            result = preprocess_for_tts(
                "Installed 3 packages and created file app.py.",
                mode="llm",
            )
            assert "Installed" in result or "app.py" in result

    def test_empty_text_returns_empty(self):
        assert preprocess_for_tts("", mode="heuristic") == ""

    def test_unknown_mode_passthrough(self):
        text = "some text"
        result = preprocess_for_tts(text, mode="unknown")
        assert result == text

    def test_truncates_long_input(self):
        text = "A" * 5000
        result = preprocess_for_tts(text, mode="off", max_len=100)
        # mode="off" still truncates to max_len
        assert len(result) <= 100

    def test_failure_callback_invoked_on_llm_error(self):
        callback = MagicMock()
        with patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("boom")):
            preprocess_for_tts("test", mode="llm", failure_callback=callback)
            callback.assert_called_once()
            args = callback.call_args[0]
            assert args[0] == "tts_preprocess"
            assert isinstance(args[1], RuntimeError)
