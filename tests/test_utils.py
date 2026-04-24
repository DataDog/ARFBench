# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
Unit tests for ARFBench utility functions.
"""

import pytest
import csv
import json
from pathlib import Path
from utils.inference_utils import parse_response
import pandas as pd

from utils.compute_statistics import (
    apply_binning,
    bin_anomaly_identification,
    bin_anomaly_magnitude,
    bin_anomaly_start,
    bin_anomaly_end,
    bin_anomaly_categorization,
    compute_multiclass_f1_metrics,
    compute_overall_weighted_f1,
    compute_statistics,
    compute_tier_weighted_f1,
    count_channels,
    parse_options,
)
from utils.inference_utils import shuffle_options


class TestParseResponse:
    """Test cases for the parse_response function."""

    CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "arfbench-qa.csv"
    _CORRECT_ANSWERS = None

    @classmethod
    def _correct_answer_set(cls) -> set[str]:
        """Load all benchmark correct answers for test fixtures."""
        if cls._CORRECT_ANSWERS is None:
            answers = set()
            with cls.CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    answer = str(row.get("correct_answer", "")).strip()
                    if answer:
                        answers.add(answer)
            cls._CORRECT_ANSWERS = answers
        return cls._CORRECT_ANSWERS

    def _csv_answer(self, answer: str) -> str:
        """Assert answer comes from benchmark correct_answer column."""
        assert answer in self._correct_answer_set()
        return answer

    def test_json_format_valid(self):
        """Test parsing valid JSON format responses."""
        anomaly_yes = self._csv_answer("Yes, there is an anomaly in the time-series")
        trend_shift = self._csv_answer("Change in Trend")
        no_anomaly = self._csv_answer("No Anomaly")

        # Test basic JSON format
        prediction = json.dumps(
            {
                "reasoning": "The metric spikes and remains elevated.",
                "answer": anomaly_yes,
            }
        )
        answer, reasoning = parse_response(prediction)
        assert answer == anomaly_yes
        assert reasoning == "The metric spikes and remains elevated."

        # Test JSON with additional fields
        prediction = json.dumps(
            {
                "reasoning": "The slope changes versus the baseline",
                "answer": trend_shift,
                "confidence": 0.9,
            }
        )
        answer, reasoning = parse_response(prediction)
        assert answer == trend_shift
        assert reasoning == "The slope changes versus the baseline"

        # Test JSON with only answer
        prediction = f'{{"answer": "{no_anomaly}"}}'
        answer, reasoning = parse_response(prediction)
        assert answer == no_anomaly
        assert reasoning is None

        # Test JSON with only reasoning
        prediction = '{"reasoning": "This is because of the pattern"}'
        answer, reasoning = parse_response(prediction)
        assert answer is None
        assert reasoning == "This is because of the pattern"

    def test_json_format_embedded(self):
        """Test parsing JSON embedded in other text."""
        level_shift = self._csv_answer("Level Shift")
        no_anomaly = self._csv_answer("No Anomaly")
        trend_shift = self._csv_answer("Change in Trend")

        prediction = (
            f"Let me analyze this. "
            f'{json.dumps({"reasoning": "Clear upward trend", "answer": level_shift})} '
            "That concludes my analysis."
        )
        answer, reasoning = parse_response(prediction)
        assert answer == level_shift
        assert reasoning == "Clear upward trend"

        # Current behavior: malformed leading braces force sentence fallback.
        prediction = (
            f"Invalid {{json}} here. "
            f'{json.dumps({"reasoning": "Valid", "answer": no_anomaly})} '
            f'{json.dumps({"answer": trend_shift})}'
        )
        answer, reasoning = parse_response(prediction)
        assert answer == "Invalid {json} here"
        assert f'"answer": "{no_anomaly}"' in reasoning
        assert f'"answer": "{trend_shift}"' in reasoning

    def test_json_format_invalid(self):
        """Test handling of invalid JSON format."""
        no_anomaly = self._csv_answer("No Anomaly")

        # Malformed JSON should fall back to other strategies
        prediction = '{"reasoning": "Missing quote, "answer": "' + no_anomaly + '"}'
        answer, reasoning = parse_response(prediction)
        # Should fall back to other parsing strategies
        assert answer is not None  # Will be parsed by other methods

        # Empty JSON
        prediction = "{}"
        answer, reasoning = parse_response(prediction)
        assert answer is None
        assert reasoning is None

    def test_answer_prefix_format(self):
        """Test parsing Answer: prefix format responses."""
        anomaly_yes = self._csv_answer("Yes, there is an anomaly in the time-series")
        no_anomaly = self._csv_answer("No Anomaly")
        trend_shift = self._csv_answer("Change in Trend")
        end_timestamp = self._csv_answer("2025-03-07 18:19:10")

        # Basic Answer: format
        prediction = (
            f"Answer: {anomaly_yes}\n"
            "The data shows a clear upward deviation over time."
        )
        answer, reasoning = parse_response(prediction)
        assert answer == anomaly_yes
        assert reasoning == "The data shows a clear upward deviation over time."

        # Answer: without reasoning
        prediction = f"Answer: {no_anomaly}"
        answer, reasoning = parse_response(prediction)
        assert answer == no_anomaly
        assert reasoning is None

        # Answer: with multiline reasoning
        prediction = (
            f"Answer: {trend_shift}\n" "Line 1 of reasoning\n" "Line 2 of reasoning"
        )
        answer, reasoning = parse_response(prediction)
        assert answer == trend_shift
        assert reasoning == "Line 1 of reasoning\nLine 2 of reasoning"

        # Answer: with extra whitespace
        prediction = (
            f"Answer:   {end_timestamp}   \n"
            "   The anomaly recovery aligns with this timestamp   "
        )
        answer, reasoning = parse_response(prediction)
        assert answer == end_timestamp
        assert reasoning == "The anomaly recovery aligns with this timestamp"

    def test_sentence_format(self):
        """Test parsing simple sentence format (answer before first period)."""
        no_anomaly = self._csv_answer("No Anomaly")
        trend_shift = self._csv_answer("Change in Trend")
        anomaly_yes = self._csv_answer("Yes, there is an anomaly in the time-series")
        not_resolved = self._csv_answer("Not resolved")

        # Basic sentence format
        prediction = f"{no_anomaly}. This option indicates no incident."
        answer, reasoning = parse_response(prediction)
        assert answer == no_anomaly
        assert reasoning == "This option indicates no incident."

        # Single answer phrase
        prediction = f"{trend_shift}. Clearly the best choice."
        answer, reasoning = parse_response(prediction)
        assert answer == trend_shift
        assert reasoning == "Clearly the best choice."

        # Longer answer before period
        prediction = f"{anomaly_yes}. The data supports this conclusion."
        answer, reasoning = parse_response(prediction)
        assert answer == anomaly_yes
        assert reasoning == "The data supports this conclusion."

        # Period without reasoning
        prediction = f"{not_resolved}."
        answer, reasoning = parse_response(prediction)
        assert answer == not_resolved
        assert reasoning is None

    def test_raw_prediction_fallback(self):
        """Test fallback to raw prediction when no other format matches."""
        no_anomaly = self._csv_answer("No Anomaly")
        trend_shift = self._csv_answer("Change in Trend")
        magnitude = self._csv_answer("100")

        # Simple text without structured format
        prediction = no_anomaly
        answer, reasoning = parse_response(prediction)
        assert answer == no_anomaly
        assert reasoning is None

        # Another plain benchmark answer
        prediction = trend_shift
        answer, reasoning = parse_response(prediction)
        assert answer == trend_shift
        assert reasoning is None

        # Numeric benchmark answer
        prediction = magnitude
        answer, reasoning = parse_response(prediction)
        assert answer == magnitude
        assert reasoning is None

    def test_edge_cases(self):
        """Test edge cases and invalid inputs."""
        # Empty string
        answer, reasoning = parse_response("")
        assert answer is None
        assert reasoning is None

        # Whitespace only
        answer, reasoning = parse_response("   \n\t   ")
        assert answer is None
        assert reasoning is None

        # None input
        answer, reasoning = parse_response(None)
        assert answer is None
        assert reasoning is None

        # Non-string input
        answer, reasoning = parse_response(123)
        assert answer is None
        assert reasoning is None

    def test_strategy_precedence(self):
        """Test that parsing strategies are applied in the correct order."""
        trend_shift = self._csv_answer("Change in Trend")
        no_anomaly = self._csv_answer("No Anomaly")

        # JSON should take precedence over Answer: format
        prediction = (
            f"Answer: {no_anomaly}\n"
            '{"reasoning": "JSON wins", '
            f'"answer": "{trend_shift}"'
            "}"
        )
        answer, reasoning = parse_response(prediction)
        assert answer == trend_shift
        assert reasoning == "JSON wins"

        # Answer: should take precedence over sentence format
        prediction = (
            f"Answer: {no_anomaly}\n"
            "Change in Trend. This period should not be used for parsing."
        )
        answer, reasoning = parse_response(prediction)
        assert answer == no_anomaly
        assert (
            reasoning == "Change in Trend. This period should not be used for parsing."
        )

    def test_realistic_model_responses(self):
        """Test with realistic model response examples."""
        anomaly_yes = self._csv_answer("Yes, there is an anomaly in the time-series")
        trend_shift = self._csv_answer("Change in Trend")
        not_resolved = self._csv_answer("Not resolved")

        # ChatGPT-style response
        prediction = (
            "Based on the time series data, I can see a clear "
            "anomaly at the end.\n\n"
            f"Answer: {anomaly_yes}\n\n"
            "The reasoning "
            "is that there's a significant spike in the last "
            "portion of the data that deviates from the normal "
            "pattern."
        )
        answer, reasoning = parse_response(prediction)
        assert answer == anomaly_yes
        assert "reasoning is that there's a significant spike" in reasoning

        # Claude-style JSON response
        claude_payload = json.dumps(
            {
                "reasoning": (
                    "The data shows a gradual decline followed by a sudden drop, "
                    "which indicates an anomaly in the system behavior."
                ),
                "answer": trend_shift,
            },
            indent=4,
        )
        prediction = (
            "Looking at this time series, I need to identify the "
            "anomaly pattern.\n\n"
            f"{claude_payload}"
        )
        answer, reasoning = parse_response(prediction)
        assert answer == trend_shift
        assert "gradual decline followed by a sudden drop" in reasoning

        # Simple direct answer
        prediction = (
            f"{not_resolved}. The anomaly is still present at the end "
            "of the observation window."
        )
        answer, reasoning = parse_response(prediction)
        assert answer == not_resolved
        assert (
            reasoning.strip()
            == "The anomaly is still present at the end of the observation window."
        )


class TestBinningFunctions:
    """Test binning helpers with real option sets from benchmark CSV."""

    CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "arfbench-qa.csv"

    @classmethod
    def _load_options_for_category(
        cls, category: str, required_options: list[str]
    ) -> tuple[str, list[str]]:
        """Load a row by category and required option values."""
        with cls.CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("task_category") != category:
                    continue

                try:
                    options = json.loads(row["options_str"])
                except (KeyError, json.JSONDecodeError, TypeError):
                    continue

                if not isinstance(options, list):
                    continue

                normalized = [str(option).strip() for option in options]
                if all(req in normalized for req in required_options):
                    return row.get("question", ""), normalized

        raise AssertionError(
            f"No {category} row found with required options: {required_options}"
        )

    def test_bin_anomaly_start_from_benchmark_question(self):
        required = [
            "Before the earliest timestamp",
            "No Anomaly",
            "2025-03-07 18:05:00",
            "2025-03-07 18:10:00",
            "2025-03-07 18:15:00",
        ]
        question, options = self._load_options_for_category("Anomaly Start", required)
        assert "start time" in question.lower()

        expected_bins = {
            "Before the earliest timestamp": "earliest",
            "No Anomaly": "no_anomaly",
            "2025-03-07 18:05:00": "early",
            "2025-03-07 18:10:00": "medium",
            "2025-03-07 18:15:00": "late",
        }
        for answer_choice, expected_bin in expected_bins.items():
            assert bin_anomaly_start(answer_choice, options) == expected_bin

    def test_bin_anomaly_end_from_benchmark_question(self):
        required = [
            "No Anomaly",
            "Not resolved",
            "2025-03-07 18:19:10",
            "2025-03-07 18:19:43",
            "2025-03-07 18:24:05",
        ]
        question, options = self._load_options_for_category("Anomaly End", required)
        assert "recovered" in question.lower()

        expected_bins = {
            "No Anomaly": "no_anomaly",
            "Not resolved": "latest",
            "2025-03-07 18:19:10": "early",
            "2025-03-07 18:19:43": "medium",
            "2025-03-07 18:24:05": "late",
        }
        for answer_choice, expected_bin in expected_bins.items():
            assert bin_anomaly_end(answer_choice, options) == expected_bin

    def test_bin_anomaly_magnitude_from_benchmark_question(self):
        required = ["No Anomaly", "0.1", "25", "50", "100"]
        question, options = self._load_options_for_category(
            "Anomaly Magnitude", required
        )
        assert "deviate" in question.lower()

        expected_bins = {
            "No Anomaly": "no_anomaly",
            "0.1": "smallest",
            "25": "small",
            "50": "medium",
            "100": "large",
        }
        for answer_choice, expected_bin in expected_bins.items():
            assert bin_anomaly_magnitude(answer_choice, options) == expected_bin

    def test_bin_anomaly_categorization_from_benchmark_question(self):
        required = [
            "Change in Trend",
            "No Anomaly",
            "Transient Spike",
            "Change in Seasonality",
            "Level Shift",
        ]
        question, options = self._load_options_for_category(
            "Anomaly Categorization", required
        )
        assert "what type of anomaly" in question.lower()

        expected_bins = {
            "Change in Trend": "Change in Trend",
            "No Anomaly": "No Anomaly",
            "Transient Spike": "Transient Spike",
            "Change in Seasonality": "Change in Seasonality/Variance",
            "Level Shift": "Level Shift",
        }
        for answer_choice, expected_bin in expected_bins.items():
            assert bin_anomaly_categorization(answer_choice, options) == expected_bin

    def test_bin_anomaly_identification_from_benchmark_question(self):
        required = [
            "No anomaly among the listed channels",
            "service:2,track_type:15",
            "service:1,track_type:41",
            "service:2,track_type:15,service:3,track_type:47",
            "service:2,track_type:15,service:1,track_type:41,service:3,track_type:47",
        ]
        question, options = self._load_options_for_category(
            "Anomaly Identification", required
        )
        assert "which channels are exhibiting anomalies" in question.lower()

        expected_bins = {
            "No anomaly among the listed channels": "no_anomaly",
            "service:1,track_type:41": "one_channel_small",
            "service:2,track_type:15": "one_channel_large",
            "service:2,track_type:15,service:3,track_type:47": "two_channel",
            "service:2,track_type:15,service:1,track_type:41,service:3,track_type:47": (
                "three_channel"
            ),
        }
        for answer_choice, expected_bin in expected_bins.items():
            assert bin_anomaly_identification(answer_choice, options) == expected_bin


class TestParseOptions:
    """Test parse_options handles both dict and plain string list formats."""

    CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "arfbench-qa.csv"

    def test_dict_format(self):
        """Test parsing [{"value": "opt1"}, {"value": "opt2"}] format."""
        options_str = '[{"value":"Yes"},{"value":"No"}]'
        result = parse_options(options_str)
        assert result == ["Yes", "No"]

    def test_plain_string_list_format(self):
        """Test parsing ["opt1", "opt2"] format."""
        options_str = '["Yes", "No"]'
        result = parse_options(options_str)
        assert result == ["Yes", "No"]

    def test_shuffled_options_are_parseable(self):
        """Test that shuffle_options output is parseable by parse_options."""
        with self.CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                shuffled = shuffle_options(row["options_str"])
                result = parse_options(shuffled)
                assert len(result) >= 2, (
                    f"parse_options returned {len(result)} options for "
                    f"shuffled options_str in row {row.get('Unnamed: 0', '?')}"
                )
                break  # one row is sufficient to verify the format

    def test_binning_works_with_shuffled_options(self):
        """Test that binning functions produce valid bins with shuffled options."""
        with self.CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["task_category"] != "Anomaly Magnitude":
                    continue
                shuffled = shuffle_options(row["options_str"])
                opts = parse_options(shuffled)
                result = bin_anomaly_magnitude(row["correct_answer"], opts)
                assert result is not None, (
                    f"bin_anomaly_magnitude returned None for correct_answer "
                    f"'{row['correct_answer']}' with shuffled options"
                )
                break

    def test_edge_cases(self):
        """Test edge cases return empty list."""
        assert parse_options("") == []
        assert parse_options("null") == []
        assert parse_options("{}") == []
        assert parse_options('"just a string"') == []
        assert parse_options("not json") == []


class TestBenchmarkDataIntegrity:
    """Validate that benchmark CSV data is internally consistent."""

    CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "arfbench-qa.csv"
    INDICATOR_NO_ANOMALY = "No anomaly in one or both series"
    IDENTIFICATION_NO_ANOMALY = "No anomaly among the listed channels"

    @classmethod
    def _load_rows(cls) -> list[dict[str, str]]:
        with cls.CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _presence_by_group(rows: list[dict[str, str]]) -> dict[str, str]:
        return {
            row["query_group"].strip(): str(row["correct_answer"]).strip()
            for row in rows
            if row["task_category"] == "Anomaly Presence"
            and "," not in row["query_group"]
        }

    def test_correct_answer_exists_in_options(self):
        """Every correct_answer must appear in its options_str list."""
        with self.CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_num, row in enumerate(reader):
                opts = json.loads(row["options_str"])
                opt_values = [str(o).strip() for o in opts]
                ca = str(row["correct_answer"]).strip()
                assert ca in opt_values, (
                    f"Row {row_num}: correct_answer {repr(ca)} "
                    f"not found in options_str {opt_values}"
                )

    def test_single_series_categories_consistent_with_presence(self):
        """
        For single-series categories other than Identification, answers should
        be Presence-consistent when Presence exists for the same query_group.
        """
        rows = self._load_rows()
        presence_by_group = self._presence_by_group(rows)
        categories_to_check = {
            "Anomaly Start",
            "Anomaly End",
            "Anomaly Magnitude",
            "Anomaly Categorization",
        }

        contradictions = []
        for row in rows:
            category = row["task_category"]
            if category not in categories_to_check:
                continue

            query_group = row["query_group"].strip()
            if "," in query_group or query_group not in presence_by_group:
                continue

            presence_answer = presence_by_group[query_group]
            category_answer = str(row["correct_answer"]).strip()
            presence_has_anomaly = presence_answer.startswith("Yes,")
            category_is_no_anomaly = category_answer == "No Anomaly"

            if presence_has_anomaly and category_is_no_anomaly:
                contradictions.append(
                    (
                        row.get("Unnamed: 0", "?"),
                        query_group,
                        category,
                        presence_answer,
                        category_answer,
                    )
                )
            if (not presence_has_anomaly) and (not category_is_no_anomaly):
                contradictions.append(
                    (
                        row.get("Unnamed: 0", "?"),
                        query_group,
                        category,
                        presence_answer,
                        category_answer,
                    )
                )

        assert not contradictions, (
            "Single-series Presence contradictions found "
            "(excluding Identification). "
            "Entries: "
            f"{contradictions}"
        )

    def test_presence_identification_cross_consistency_when_available(self):
        """
        Cross-consistency between Presence and Identification for single-series
        rows with both labels available.

        - If Presence says no anomaly, Identification must use the
          identification no-anomaly label.
        - If Identification chooses one or more anomalous channels, Presence
          must indicate anomaly.
        """
        rows = self._load_rows()
        presence_by_group = self._presence_by_group(rows)

        contradictions = []
        for row in rows:
            if row["task_category"] != "Anomaly Identification":
                continue

            query_group = row["query_group"].strip()
            if "," in query_group or query_group not in presence_by_group:
                continue

            presence_answer = presence_by_group[query_group]
            identification_answer = str(row["correct_answer"]).strip()
            presence_has_anomaly = presence_answer.startswith("Yes,")
            identification_is_no_anomaly = (
                identification_answer == self.IDENTIFICATION_NO_ANOMALY
            )

            if (not presence_has_anomaly) and (not identification_is_no_anomaly):
                contradictions.append(
                    (
                        row.get("Unnamed: 0", "?"),
                        query_group,
                        presence_answer,
                        identification_answer,
                    )
                )

            if presence_has_anomaly and identification_is_no_anomaly:
                # This case is allowed because Identification is over only the
                # listed channel subset.
                continue

        assert not contradictions, (
            "Presence/Identification cross-consistency violations found. "
            "Entries: "
            f"{contradictions}"
        )

    def test_correlation_consistency_with_presence_when_available(self):
        """
        For paired rows where both component series have Presence labels, the
        Correlation answer must agree with those Presence states.
        """
        rows = self._load_rows()
        presence_by_group = self._presence_by_group(rows)

        contradictions = []
        for row in rows:
            if row["task_category"] != "Anomaly Correlation":
                continue

            pair = row["query_group"].strip()
            groups = [group.strip() for group in pair.split(",")]
            if len(groups) != 2:
                continue
            if groups[0] not in presence_by_group or groups[1] not in presence_by_group:
                continue

            answer = str(row["correct_answer"]).strip()
            first_has_anomaly = presence_by_group[groups[0]].startswith("Yes,")
            second_has_anomaly = presence_by_group[groups[1]].startswith("Yes,")

            if first_has_anomaly and second_has_anomaly:
                valid_answers = {
                    "Yes, there is an anomaly in both and they are correlated",
                    "No, there is an anomaly in both but they are not correlated",
                }
            elif first_has_anomaly and (not second_has_anomaly):
                valid_answers = {"No, there is an anomaly only in time-series 1"}
            elif (not first_has_anomaly) and second_has_anomaly:
                valid_answers = {"No, there is an anomaly only in time-series 2"}
            else:
                valid_answers = {"No, there is no anomaly in either time-series"}

            if answer not in valid_answers:
                contradictions.append(
                    (
                        row.get("Unnamed: 0", "?"),
                        pair,
                        answer,
                        presence_by_group[groups[0]],
                        presence_by_group[groups[1]],
                    )
                )

        assert not contradictions, (
            "Correlation/Presence contradictions found. "
            "Entries: "
            f"{contradictions}"
        )

    def test_indicator_requires_no_anomaly_if_either_series_has_no_anomaly(self):
        """
        Indicator should use its no-anomaly label whenever at least one
        component series is labeled as no anomaly by Presence.
        """
        rows = self._load_rows()
        presence_by_group = self._presence_by_group(rows)

        contradictions = []
        for row in rows:
            if row["task_category"] != "Anomaly Indicator":
                continue

            pair = row["query_group"].strip()
            groups = [group.strip() for group in pair.split(",")]
            if len(groups) != 2:
                continue
            if groups[0] not in presence_by_group or groups[1] not in presence_by_group:
                continue

            first_has_anomaly = presence_by_group[groups[0]].startswith("Yes,")
            second_has_anomaly = presence_by_group[groups[1]].startswith("Yes,")
            answer = str(row["correct_answer"]).strip()

            if (
                not first_has_anomaly or not second_has_anomaly
            ) and answer != self.INDICATOR_NO_ANOMALY:
                contradictions.append(
                    (
                        row.get("Unnamed: 0", "?"),
                        pair,
                        answer,
                        presence_by_group[groups[0]],
                        presence_by_group[groups[1]],
                    )
                )

        assert not contradictions, (
            "Indicator/Presence contradictions found. " "Entries: " f"{contradictions}"
        )

    def test_presence_indicator_cross_consistency_when_both_have_anomaly(self):
        """
        If both component series are anomalous by Presence, Indicator should
        not use the indicator no-anomaly label.
        """
        rows = self._load_rows()
        presence_by_group = self._presence_by_group(rows)

        contradictions = []
        for row in rows:
            if row["task_category"] != "Anomaly Indicator":
                continue

            pair = row["query_group"].strip()
            groups = [group.strip() for group in pair.split(",")]
            if len(groups) != 2:
                continue
            if groups[0] not in presence_by_group or groups[1] not in presence_by_group:
                continue

            first_has_anomaly = presence_by_group[groups[0]].startswith("Yes,")
            second_has_anomaly = presence_by_group[groups[1]].startswith("Yes,")
            answer = str(row["correct_answer"]).strip()

            if (
                first_has_anomaly
                and second_has_anomaly
                and answer == self.INDICATOR_NO_ANOMALY
            ):
                contradictions.append(
                    (
                        row.get("Unnamed: 0", "?"),
                        pair,
                        answer,
                        presence_by_group[groups[0]],
                        presence_by_group[groups[1]],
                    )
                )

        assert not contradictions, (
            "Presence/Indicator cross-consistency violations found "
            "for both-anomalous pairs. Entries: "
            f"{contradictions}"
        )

    def test_indicator_consistency_with_correlation_on_shared_pairs(self):
        """
        When both Correlation and Indicator exist for the same pair:
        - if Correlation says anomalies are not present in both series,
          Indicator must use the indicator no-anomaly label
        - if Correlation says anomalies are in both but not correlated,
          Indicator must be the explicit "not correlated" class
        """
        rows = self._load_rows()

        correlation_by_pair = {}
        indicator_by_pair = {}
        for row in rows:
            pair = row["query_group"].strip()
            answer = str(row["correct_answer"]).strip()
            row_id = row.get("Unnamed: 0", "?")
            if row["task_category"] == "Anomaly Correlation":
                correlation_by_pair[pair] = (answer, row_id)
            elif row["task_category"] == "Anomaly Indicator":
                indicator_by_pair[pair] = (answer, row_id)

        no_shared_anomaly_answers = {
            "No, there is no anomaly in either time-series",
            "No, there is an anomaly only in time-series 1",
            "No, there is an anomaly only in time-series 2",
        }
        not_correlated_answer = (
            "The anomaly in time-series 1 is not correlated to the "
            "anomaly in time-series 2."
        )

        contradictions = []
        for pair in sorted(set(correlation_by_pair).intersection(indicator_by_pair)):
            correlation_answer, correlation_row_id = correlation_by_pair[pair]
            indicator_answer, indicator_row_id = indicator_by_pair[pair]

            if (
                correlation_answer in no_shared_anomaly_answers
                and indicator_answer != self.INDICATOR_NO_ANOMALY
            ):
                contradictions.append(
                    (
                        pair,
                        correlation_row_id,
                        indicator_row_id,
                        correlation_answer,
                        indicator_answer,
                    )
                )

            if (
                correlation_answer
                == "No, there is an anomaly in both but they are not correlated"
                and indicator_answer != not_correlated_answer
            ):
                contradictions.append(
                    (
                        pair,
                        correlation_row_id,
                        indicator_row_id,
                        correlation_answer,
                        indicator_answer,
                    )
                )

        assert not contradictions, (
            "Correlation/Indicator contradictions found. "
            "Entries: "
            f"{contradictions}"
        )


# ---------------------------------------------------------------------------
# Helper: build a minimal results DataFrame for metric tests
# ---------------------------------------------------------------------------


def _make_results_df(rows):
    """Build a results DataFrame from (category, correct, model) tuples."""
    records = []
    for i, (cat, correct, model) in enumerate(rows):
        opts = json.dumps([correct, model] if correct != model else [correct, "other"])
        records.append(
            {
                "run": 1,
                "model": "test-model",
                "query_group": f"q{i}",
                "question": "placeholder",
                "task_category": cat,
                "correct_answer": correct,
                "model_answer": model,
                "is_correct": correct == model,
                "options": opts,
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tests: count_channels
# ---------------------------------------------------------------------------


class TestCountChannels:

    def test_single_channel(self):
        assert count_channels("pod_name:614") == 1

    def test_and_delimiter(self):
        assert count_channels("method:1 and method:2") == 2
        assert count_channels("a:1 and b:2 and c:3") == 3

    def test_comma_space_delimiter(self):
        assert count_channels("pod_name:614, pod_name:198") == 2

    def test_repeating_dimension(self):
        assert count_channels("service:2,track_type:15,service:1,track_type:41") == 2

    def test_no_anomaly(self):
        assert count_channels("No Anomaly") == 1


# ---------------------------------------------------------------------------
# Tests: apply_binning
# ---------------------------------------------------------------------------


class TestApplyBinning:

    def test_routes_presence(self):
        assert apply_binning("Yes", [], "Anomaly Presence") == "Yes"

    def test_routes_categorization(self):
        assert (
            apply_binning("Level Shift", [], "Anomaly Categorization") == "Level Shift"
        )

    def test_routes_magnitude(self):
        opts = ["No Anomaly", "1", "10", "100", "1000"]
        assert apply_binning("1", opts, "Anomaly Magnitude") == "smallest"
        assert apply_binning("1000", opts, "Anomaly Magnitude") == "large"

    def test_routes_start(self):
        opts = ["No Anomaly", "2025-03-07 10:00:00", "2025-03-07 11:00:00"]
        assert apply_binning("2025-03-07 10:00:00", opts, "Anomaly Start") == "earliest"

    def test_routes_end(self):
        opts = ["No Anomaly", "2025-03-07 10:00:00", "2025-03-07 11:00:00"]
        assert apply_binning("2025-03-07 11:00:00", opts, "Anomaly End") == "latest"

    def test_routes_identification(self):
        opts = ["No Anomaly", "pod:1", "pod:2"]
        assert (
            apply_binning("No Anomaly", opts, "Anomaly Identification") == "no_anomaly"
        )

    def test_none_for_empty_answer(self):
        assert apply_binning("", [], "Anomaly Presence") is None

    def test_correlation_passthrough(self):
        val = "Yes, there is an anomaly in both"
        assert apply_binning(val, [], "Anomaly Correlation") == val

    def test_indicator_passthrough(self):
        val = "Leading indicator"
        assert apply_binning(val, [], "Anomaly Indicator") == val


# ---------------------------------------------------------------------------
# Tests: compute_statistics
# ---------------------------------------------------------------------------


class TestComputeStatistics:

    def test_per_category_accuracy(self):
        df = _make_results_df(
            [
                ("Anomaly Presence", "Yes", "Yes"),
                ("Anomaly Presence", "No", "No"),
                ("Anomaly Presence", "Yes", "No"),
            ]
        )
        stats = compute_statistics(df)
        row = stats[stats["task_category"] == "Anomaly Presence"].iloc[0]
        assert abs(row["accuracy"] - 66.666) < 1  # 2/3

    def test_tier_accuracies_present(self):
        df = _make_results_df(
            [
                ("Anomaly Presence", "Yes", "Yes"),
                ("Anomaly Correlation", "Yes", "No"),
            ]
        )
        stats = compute_statistics(df)
        assert "tier_1_accuracy" in stats.columns
        assert "tier_3_accuracy" in stats.columns

    def test_empty_df(self):
        stats = compute_statistics(pd.DataFrame())
        assert stats.empty

    def test_only_first_run(self):
        """Only run=1 should be used for accuracy."""
        df = _make_results_df(
            [
                ("Anomaly Presence", "Yes", "Yes"),
            ]
        )
        # Add a second run where model is wrong
        run2 = df.copy()
        run2["run"] = 2
        run2["model_answer"] = "No"
        run2["is_correct"] = False
        combined = pd.concat([df, run2], ignore_index=True)
        stats = compute_statistics(combined)
        row = stats[stats["task_category"] == "Anomaly Presence"].iloc[0]
        assert row["accuracy"] == 100.0  # only run 1 counts


# ---------------------------------------------------------------------------
# Tests: compute_multiclass_f1_metrics
# ---------------------------------------------------------------------------


class TestComputeMulticlassF1:

    def test_perfect_predictions(self):
        df = _make_results_df(
            [
                ("Anomaly Presence", "Yes", "Yes"),
                ("Anomaly Presence", "No", "No"),
            ]
        )
        f1_df = compute_multiclass_f1_metrics(df)
        cat_row = f1_df[f1_df["task_category"] == "Anomaly Presence"].iloc[0]
        assert cat_row["macro_f1"] == 1.0

    def test_overall_row_present(self):
        df = _make_results_df(
            [
                ("Anomaly Presence", "Yes", "Yes"),
                ("Anomaly Correlation", "No", "No"),
            ]
        )
        f1_df = compute_multiclass_f1_metrics(df)
        overall = f1_df[f1_df["task_category"] == "Overall"]
        assert len(overall) == 1
        assert overall.iloc[0]["macro_f1"] == 1.0

    def test_overall_is_weighted_average(self):
        """Overall F1 should weight by n_samples, not simple average."""
        # 4 correct Presence, 1 wrong Correlation → overall should lean
        # towards the larger category.
        df = _make_results_df(
            [
                ("Anomaly Presence", "Yes", "Yes"),
                ("Anomaly Presence", "Yes", "Yes"),
                ("Anomaly Presence", "No", "No"),
                ("Anomaly Presence", "No", "No"),
                ("Anomaly Correlation", "Yes", "No"),
            ]
        )
        f1_df = compute_multiclass_f1_metrics(df)
        overall = f1_df[f1_df["task_category"] == "Overall"].iloc[0]
        presence = f1_df[f1_df["task_category"] == "Anomaly Presence"].iloc[0]
        correlation = f1_df[f1_df["task_category"] == "Anomaly Correlation"].iloc[0]

        expected = (4 * presence["macro_f1"] + 1 * correlation["macro_f1"]) / (4 + 1)
        assert abs(overall["macro_f1"] - expected) < 1e-9

    def test_empty_df(self):
        f1_df = compute_multiclass_f1_metrics(pd.DataFrame())
        assert f1_df.empty

    def test_invalid_model_answer_counted_as_wrong(self):
        """None model answers should not produce F1=1."""
        df = _make_results_df(
            [
                ("Anomaly Presence", "Yes", "Yes"),
                ("Anomaly Presence", "No", None),
            ]
        )
        df.at[1, "model_answer"] = None
        f1_df = compute_multiclass_f1_metrics(df)
        cat_row = f1_df[f1_df["task_category"] == "Anomaly Presence"].iloc[0]
        assert cat_row["macro_f1"] < 1.0

    def test_seed_makes_invalid_mapping_deterministic(self):
        """Same seed should produce identical metrics across repeated calls."""
        df = _make_results_df(
            [
                ("Anomaly Presence", "Yes", "Yes"),
                ("Anomaly Presence", "No", "No"),
                ("Anomaly Presence", "Yes", "INVALID_OOV_LABEL"),
                ("Anomaly Presence", "No", "INVALID_OOV_LABEL"),
                ("Anomaly Presence", "Yes", "INVALID_OOV_LABEL"),
                ("Anomaly Presence", "No", "INVALID_OOV_LABEL"),
            ]
        )
        f1_a = compute_multiclass_f1_metrics(df, random_seed=123)
        f1_b = compute_multiclass_f1_metrics(df, random_seed=123)
        row_a = f1_a[f1_a["task_category"] == "Anomaly Presence"].iloc[0]
        row_b = f1_b[f1_b["task_category"] == "Anomaly Presence"].iloc[0]
        assert row_a["macro_f1"] == row_b["macro_f1"]
        assert row_a["macro_precision"] == row_b["macro_precision"]
        assert row_a["macro_recall"] == row_b["macro_recall"]


# ---------------------------------------------------------------------------
# Tests: compute_tier_weighted_f1
# ---------------------------------------------------------------------------


class TestComputeTierWeightedF1:

    def test_tiers_present(self):
        df = _make_results_df(
            [
                ("Anomaly Presence", "Yes", "Yes"),
                ("Anomaly Start", "No Anomaly", "No Anomaly"),
                ("Anomaly Correlation", "Yes", "Yes"),
            ]
        )
        tier_df = compute_tier_weighted_f1(df)
        tiers = set(tier_df["tier"])
        assert {"Tier 1", "Tier 2", "Tier 3", "Overall"} == tiers

    def test_overall_row_present(self):
        df = _make_results_df(
            [
                ("Anomaly Presence", "Yes", "Yes"),
            ]
        )
        tier_df = compute_tier_weighted_f1(df)
        overall = tier_df[tier_df["tier"] == "Overall"]
        assert len(overall) == 1

    def test_overall_weighted_across_tiers(self):
        df = _make_results_df(
            [
                ("Anomaly Presence", "Yes", "Yes"),
                ("Anomaly Presence", "No", "No"),
                ("Anomaly Correlation", "Yes", "No"),
            ]
        )
        tier_df = compute_tier_weighted_f1(df)
        t1 = tier_df[tier_df["tier"] == "Tier 1"].iloc[0]
        t3 = tier_df[tier_df["tier"] == "Tier 3"].iloc[0]
        overall = tier_df[tier_df["tier"] == "Overall"].iloc[0]
        expected = (2 * t1["weighted_f1"] + 1 * t3["weighted_f1"]) / (2 + 1)
        assert abs(overall["weighted_f1"] - expected) < 1e-9

    def test_empty_df(self):
        tier_df = compute_tier_weighted_f1(pd.DataFrame())
        assert tier_df.empty

    def test_overall_matches_category_overall_when_reusing_f1_pass(self):
        """Tier Overall must equal category Overall when sharing one F1 pass."""
        df = _make_results_df(
            [
                ("Anomaly Presence", "Yes", "Yes"),
                ("Anomaly Presence", "No", "Yes"),
                ("Anomaly Start", "No Anomaly", "No Anomaly"),
                ("Anomaly End", "2025-03-07 18:19:10", "2025-03-07 18:19:43"),
                ("Anomaly Correlation", "Yes", "No"),
            ]
        )
        f1_df = compute_multiclass_f1_metrics(df)
        tier_df = compute_tier_weighted_f1(df, category_metrics_df=f1_df)

        cat_overall = f1_df[f1_df["task_category"] == "Overall"].iloc[0]
        tier_overall = tier_df[tier_df["tier"] == "Overall"].iloc[0]

        assert abs(tier_overall["weighted_f1"] - cat_overall["macro_f1"]) < 1e-12
        assert (
            abs(tier_overall["weighted_precision"] - cat_overall["macro_precision"])
            < 1e-12
        )
        assert abs(tier_overall["weighted_recall"] - cat_overall["macro_recall"]) < 1e-12


# ---------------------------------------------------------------------------
# Tests: compute_overall_weighted_f1
# ---------------------------------------------------------------------------


class TestComputeOverallWeightedF1:

    def test_basic(self):
        cat_df = pd.DataFrame(
            [
                {
                    "task_category": "A",
                    "macro_f1": 1.0,
                    "macro_precision": 1.0,
                    "macro_recall": 1.0,
                    "n_samples": 10,
                },
                {
                    "task_category": "B",
                    "macro_f1": 0.0,
                    "macro_precision": 0.0,
                    "macro_recall": 0.0,
                    "n_samples": 10,
                },
            ]
        )
        result = compute_overall_weighted_f1(cat_df)
        assert result is not None
        assert result["task_category"] == "Overall"
        assert abs(result["macro_f1"] - 0.5) < 1e-9

    def test_weighted_not_simple_average(self):
        cat_df = pd.DataFrame(
            [
                {
                    "task_category": "A",
                    "macro_f1": 1.0,
                    "macro_precision": 1.0,
                    "macro_recall": 1.0,
                    "n_samples": 90,
                },
                {
                    "task_category": "B",
                    "macro_f1": 0.0,
                    "macro_precision": 0.0,
                    "macro_recall": 0.0,
                    "n_samples": 10,
                },
            ]
        )
        result = compute_overall_weighted_f1(cat_df)
        assert abs(result["macro_f1"] - 0.9) < 1e-9

    def test_empty(self):
        assert compute_overall_weighted_f1(pd.DataFrame()) is None

    def test_excludes_existing_overall_row(self):
        cat_df = pd.DataFrame(
            [
                {
                    "task_category": "A",
                    "macro_f1": 1.0,
                    "macro_precision": 1.0,
                    "macro_recall": 1.0,
                    "n_samples": 10,
                },
                {
                    "task_category": "Overall",
                    "macro_f1": 0.5,
                    "macro_precision": 0.5,
                    "macro_recall": 0.5,
                    "n_samples": 20,
                },
            ]
        )
        result = compute_overall_weighted_f1(cat_df)
        # Should only use category A, not the existing Overall row
        assert abs(result["macro_f1"] - 1.0) < 1e-9
        assert result["n_samples"] == 10


if __name__ == "__main__":
    pytest.main([__file__])
