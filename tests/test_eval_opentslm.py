#!/usr/bin/env python3
# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache-2.0 License.
#
# This product includes software developed at Datadog
# (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""Tests for OpenTSLM evaluator helper logic."""

from evaluation.eval_opentslm import (
    is_prediction_correct,
    maybe_shuffle_batch_options,
    normalize_answer_text,
)


def test_normalize_answer_text_removes_eos_markers():
    raw = "No anomaly </s><|end_of_text|>  "
    assert normalize_answer_text(raw) == "No anomaly"


def test_is_prediction_correct_supports_prefix_matching():
    assert is_prediction_correct("No anomaly", "No anomaly")
    assert is_prediction_correct("service:web", "service:web,service:db")
    assert not is_prediction_correct("", "No anomaly")
    assert not is_prediction_correct("No anomaly", "")


def test_maybe_shuffle_batch_options_updates_post_prompt_consistently():
    sample = {
        "options_str": '["A","B","C","D","E"]',
        "post_prompt": (
            "Question: test?\n\nOptions: [\"A\",\"B\",\"C\",\"D\",\"E\"]\n\nAnswer:"
        ),
    }
    shuffled = maybe_shuffle_batch_options([sample], shuffle_option_order=True)[0]

    assert shuffled["options_str"] != sample["options_str"]
    assert shuffled["options_str"] in shuffled["post_prompt"]
    assert sample["options_str"] in sample["post_prompt"]


def test_maybe_shuffle_batch_options_noop_when_disabled():
    sample = {"options_str": '["A","B"]', "post_prompt": "Options: [\"A\",\"B\"]"}
    batch = [sample]
    result = maybe_shuffle_batch_options(batch, shuffle_option_order=False)
    assert result is batch
