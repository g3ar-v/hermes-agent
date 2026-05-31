"""Pre-process agent response text before TTS playback.

Produces a voice-friendly summary rather than reading the full response
verbatim. Runs synchronously (caller controls threading) and falls back
to heuristic extraction when LLM mode fails.

Usage:
    from agent.tts_preprocessor import preprocess_for_tts
    voice_text = preprocess_for_tts(response_text, mode="llm")

Config:
    hermes config set voice.tts_summarize llm       # use auxiliary LLM
    hermes config set voice.tts_summarize heuristic  # rule-based only
    hermes config set voice.tts_summarize off        # disable (default)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_TTS_PREPROCESS_PROMPT = (
    "You are an AI assistant. You have just finished speaking to the user — "
    "the text below is your own response that was displayed to them. "
    "Now you need to say it aloud to them via text-to-speech. "
    "Condense it into a short spoken summary, as if you are continuing "
    "the conversation and telling them what you just said. "
    "Speak naturally, in the first person, directly addressing the user. "
    "If tables, lists of data, or structured output were displayed, "
    "refer to them briefly by their key takeaways rather than reading every item. "
    "Skip code blocks, URLs, file paths, raw terminal output, and "
    "step-by-step instructions — just convey the essential meaning. "
    "Keep it conversational and concise: one to three sentences at most. "
    "Return ONLY the spoken summary, nothing else."
)


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------


def _build_heuristic_summary(text: str, max_len: int = 400) -> str:
    """Extract key actions into a single summary sentence.

    Looks for structured action markers the model commonly emits:
    file creation, package installs, service starts, etc.
    Falls back to the first descriptive sentence as a last resort.
    """
    actions: list[str] = []

    # File operations: "created file X", "wrote file Y"
    file_creates = re.findall(
        r"(?:created|wrote|saved|updated).*?(?:file|config|script)\s+([\S]+)",
        text,
        re.I,
    )
    if file_creates:
        actions.append(f"Created {', '.join(file_creates[:3])}")

    # Installs
    installs = re.findall(
        r"(?:installed|added)\s+(\d+)\s+packages?",
        text,
        re.I,
    )
    if installs:
        actions.append(f"Installed {installs[0]} packages")

    # Services started/stopped
    services = re.findall(
        r"(?:started|stopped|restarted)\s+the\s+(\S+)\s+service",
        text,
        re.I,
    )
    if services:
        actions.append(f"Started the {services[0]} service")

    # Run commands / tests
    ran_tests = re.findall(
        r"(?:ran|executed)\s+(\d+)\s+tests?",
        text,
        re.I,
    )
    if ran_tests:
        actions.append(f"Ran {ran_tests[0]} tests")

    if actions:
        summary = "; ".join(actions[:3])
        if not summary.endswith("."):
            summary += "."
        return summary

    # Last resort: first descriptive sentence that isn't code/markup
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for s in sentences:
        s = s.strip()
        if (
            len(s) > 10
            and len(s) < max_len
            and not s.startswith(("```", "``", "import ", "#", "|", "-"))
        ):
            # Strip residual inline markdown
            s = re.sub(r"`([^`]+)`", r"\1", s)
            s = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", s)
            return s.rstrip(".") + "."

    return text.strip()[:max_len]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def preprocess_for_tts(
    response_text: str,
    mode: str = "off",
    max_len: int = 4000,
    failure_callback=None,
) -> str:
    """Pre-process agent response for TTS playback.

    Args:
        response_text: The full agent response text.
        mode: ``"llm"`` (summarise via auxiliary model), ``"heuristic"``
              (rule-based only), or ``"off"`` (passthrough — current behaviour).
        max_len: Maximum length of the input text to process.
        failure_callback: Optional callable ``(str, Exception)`` invoked when the
              LLM call fails. Used to surface aux failures to the user.

    Returns:
        Voice-friendly summary text. Falls back to heuristic extraction
        silently when LLM mode fails.
    """
    if not response_text:
        return response_text

    # Truncate input
    text = response_text[:max_len] if len(response_text) > max_len else response_text

    if mode == "off":
        return text

    if mode == "llm":
        return _llm_summarize(text, failure_callback=failure_callback)
    elif mode == "heuristic":
        return _build_heuristic_summary(text)

    # Unknown mode: passthrough
    return text


def _llm_summarize(
    text: str,
    failure_callback=None,
) -> str:
    """Use the auxiliary LLM to produce a one-sentence summary.

    Falls back to heuristic extraction on any failure (no audio dropouts).
    """
    from agent.auxiliary_client import call_llm

    messages = [
        {"role": "system", "content": _TTS_PREPROCESS_PROMPT},
        {"role": "user", "content": text[:2000]},
    ]

    try:
        response = call_llm(
            task="tts_preprocess",
            messages=messages,
            max_tokens=150,
            temperature=0.3,
            timeout=15.0,
        )
        summary = (response.choices[0].message.content or "").strip()
        if summary:
            # Clean up common LLM artefacts
            summary = summary.strip("\"'")
            if summary.lower().startswith(("summary:", "done:", "result:")):
                summary = summary.split(":", 1)[1].strip()
            return summary
    except Exception as e:
        logger.debug("TTS LLM pre-processing failed, falling back to heuristic: %s", e)
        if failure_callback is not None:
            try:
                failure_callback("tts_preprocess", e)
            except Exception:
                logger.debug("tts_preprocess failure_callback raised", exc_info=True)

    return _build_heuristic_summary(text)
