#!/usr/bin/env python3
# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache-2.0 License.
#
# This product includes software developed at Datadog
# (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""Tests for OpenAI text-eval context overflow recovery helpers."""

from evaluation.eval_openai_text import (
    ContextLengthExceededError,
    _compute_truncation_ratio,
    _extract_context_limit_and_usage,
    _next_coarser_intervals,
    _truncate_time_series_data,
)


def test_extract_context_limit_and_usage():
    message = (
        "Error code: 400 - {'error': {'message': 'Input tokens exceed the "
        "configured limit of 272000 tokens. Your messages resulted in 331593 "
        "tokens. Please reduce the length of the messages.', "
        "'type': 'invalid_request_error', 'param': 'messages', "
        "'code': 'context_length_exceeded'}}"
    )
    limit_tokens, used_tokens = _extract_context_limit_and_usage(message)
    assert limit_tokens == 272000
    assert used_tokens == 331593


def test_next_coarser_intervals():
    assert _next_coarser_intervals(300) == [1800, 3600, 86400]
    assert _next_coarser_intervals(86400) == []
    assert _next_coarser_intervals(42) == []


def test_compute_truncation_ratio_uses_limit_and_margin():
    context_error = ContextLengthExceededError(
        message="context overflow",
        limit_tokens=272000,
        used_tokens=331593,
    )
    ratio = _compute_truncation_ratio(context_error, truncation_attempt=1)

    # Bounded by fallback ratio on first attempt and by model-derived ratio.
    assert 0.20 <= ratio <= 0.75


def test_compute_truncation_ratio_fallback_progression():
    context_error = ContextLengthExceededError(message="context overflow")
    assert (
        _compute_truncation_ratio(context_error, truncation_attempt=1) == 0.75
    )
    assert (
        _compute_truncation_ratio(context_error, truncation_attempt=2) == 0.60
    )
    assert (
        _compute_truncation_ratio(context_error, truncation_attempt=3) == 0.45
    )
    assert (
        _compute_truncation_ratio(context_error, truncation_attempt=4) == 0.30
    )
    assert (
        _compute_truncation_ratio(context_error, truncation_attempt=9) == 0.30
    )


def test_truncate_time_series_data_reduces_payload():
    original = [
        "header\n" + ("a" * 5000) + "\n" + ("z" * 5000),
        "short-series",
    ]
    truncated = _truncate_time_series_data(original, truncation_ratio=0.5)

    assert len(truncated) == 2
    assert len(truncated[0]) < len(original[0])
    assert "[...TRUNCATED FOR CONTEXT WINDOW...]" in truncated[0]

    # Very short series may remain unchanged.
    assert truncated[1] == original[1]
