# Unless explicitly stated otherwise all files in this repository
# are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog
# (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
Unit tests for evaluation scripts
"""

import ast
import csv
import json
from pathlib import Path

from utils.inference_utils import shuffle_options


def _evaluation_files() -> list[Path]:
    """Return all top-level evaluation scripts."""
    evaluation_dir = Path(__file__).resolve().parents[1] / "evaluation"
    return sorted(evaluation_dir.glob("eval_*.py"))


def _calls_name(node: ast.AST, name: str) -> bool:
    """Check whether a call expression targets a function by name."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and (node.func.id == name)
    )


def _function_def(
    tree: ast.AST, function_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return a top-level function definition by name."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                return node
    return None


def _node_references_name(node: ast.AST, name: str) -> bool:
    """Check whether a node references a given name."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == name:
            return True
    return False


def _is_row_question_subscript(node: ast.AST) -> bool:
    """Detect row['question'] or row[\"question\"] access."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "row"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "question"
    )


def _node_references_row_question(node: ast.AST) -> bool:
    """Check whether a node reads row['question']."""
    for child in ast.walk(node):
        if _is_row_question_subscript(child):
            return True
    return False


def _function_has_call_with_keyword(
    function_node: ast.AST,
    keyword_name: str,
    keyword_value_name: str | None = None,
) -> bool:
    """Detect calls with a specific keyword, optionally bound to a name."""
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue

        for keyword in node.keywords:
            if keyword.arg != keyword_name:
                continue
            if keyword_value_name is None:
                return True
            if (
                isinstance(keyword.value, ast.Name)
                and keyword.value.id == keyword_value_name
            ):
                return True
    return False


def _function_has_text_payload_with_question(function_node: ast.AST) -> bool:
    """Detect {'text': ...question...} payload dictionaries."""
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Dict):
            continue

        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "text"
                and _node_references_name(value, "question")
            ):
                return True
    return False


def _function_assigns_name_from_question(
    function_node: ast.AST,
    assigned_name: str,
    source_name: str,
) -> bool:
    """Detect assignments like assigned_name = ...source_name..."""
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Assign):
            continue

        has_target = any(
            isinstance(target, ast.Name) and target.id == assigned_name
            for target in node.targets
        )
        if has_target and _node_references_name(node.value, source_name):
            return True
    return False


def _function_calls_with_row_question(
    function_node: ast.AST, callee_name: str
) -> bool:
    """Detect function calls where any arg/kwarg references row['question']."""
    for node in ast.walk(function_node):
        if not _calls_name(node, callee_name):
            continue

        if any(_node_references_row_question(arg) for arg in node.args):
            return True
        if any(_node_references_row_question(kw.value) for kw in node.keywords):
            return True
    return False


def _function_calls_with_named_argument(
    function_node: ast.AST,
    callee_name: str,
    arg_name: str,
) -> bool:
    """Detect calls where any argument is a given variable name."""
    for node in ast.walk(function_node):
        if not _calls_name(node, callee_name):
            continue

        if any(isinstance(arg, ast.Name) and arg.id == arg_name for arg in node.args):
            return True
        if any(
            isinstance(kw.value, ast.Name) and kw.value.id == arg_name
            for kw in node.keywords
        ):
            return True
    return False


def _function_assigns_from_row_question(
    function_node: ast.AST, assigned_name: str
) -> bool:
    """Detect assigned_name = ...row['question']..."""
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Assign):
            continue

        has_target = any(
            isinstance(target, ast.Name) and target.id == assigned_name
            for target in node.targets
        )
        if has_target and _node_references_row_question(node.value):
            return True
    return False


def _function_uses_question_split_on_time_series(function_node: ast.AST) -> bool:
    """Detect question.split('Time-series...') truncation patterns."""
    for node in ast.walk(function_node):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "split"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "question"
        ):
            continue

        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            if "Time-series" in first_arg.value:
                return True
    return False


def _load_first_benchmark_question() -> str:
    """Load the first benchmark question from ARFBench CSV."""
    benchmark_path = Path(__file__).resolve().parents[1] / "data" / "arfbench-qa.csv"
    with benchmark_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader)
    return first_row["question"]


def _extract_option_values_for_shuffle(raw_options: str) -> list[str] | None:
    """Parse options text into a list of option values."""
    if not raw_options:
        return None

    try:
        parsed = json.loads(raw_options)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None

    option_values = []
    for option in parsed:
        if isinstance(option, dict):
            option_text = None
            for key in ("value", "text", "option", "answer"):
                raw_value = option.get(key)
                if raw_value is None:
                    continue
                candidate = str(raw_value).strip()
                if candidate:
                    option_text = candidate
                    break
            if option_text:
                option_values.append(option_text)
        else:
            candidate = str(option).strip()
            if candidate:
                option_values.append(candidate)

    if len(option_values) < 2:
        return None
    if len(set(option_values)) != len(option_values):
        return None
    return option_values


def _load_uniformity_question_and_options() -> tuple[str, list[str]]:
    """Select one benchmark question with parseable, unique options."""
    benchmark_path = Path(__file__).resolve().parents[1] / "data" / "arfbench-qa.csv"
    fallback_choice = None

    with benchmark_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_options = row.get("options_str") or row.get("options")
            option_values = _extract_option_values_for_shuffle(raw_options)
            if not option_values:
                continue

            question_text = row.get("question", "")
            if len(option_values) >= 4:
                return question_text, option_values
            if fallback_choice is None:
                fallback_choice = (question_text, option_values)

    if fallback_choice:
        return fallback_choice
    raise AssertionError("Could not find benchmark row with parseable options")


def _has_cli_image_dir_arg(tree: ast.AST) -> bool:
    """Detect parser.add_argument('--image-dir', ...) usage."""
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue

        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == "--image-dir":
                return True
    return False


def _has_forwarding_from_args_image_dir(tree: ast.AST) -> bool:
    """Detect image_dir=args.image_dir in function calls."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        for keyword in node.keywords:
            if (
                keyword.arg == "image_dir"
                and isinstance(keyword.value, ast.Attribute)
                and isinstance(keyword.value.value, ast.Name)
                and keyword.value.value.id == "args"
                and keyword.value.attr == "image_dir"
            ):
                return True
    return False


def _all_get_image_paths_calls_use_image_dir(tree: ast.AST) -> bool:
    """Ensure get_image_paths(..., image_dir, ...) is explicit."""
    calls = [node for node in ast.walk(tree) if _calls_name(node, "get_image_paths")]
    if not calls:
        return False

    for call in calls:
        has_positional_image_dir = (
            len(call.args) >= 2
            and isinstance(call.args[1], ast.Name)
            and call.args[1].id == "image_dir"
        )
        has_keyword_image_dir = any(
            keyword.arg == "image_dir"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "image_dir"
            for keyword in call.keywords
        )
        if not (has_positional_image_dir or has_keyword_image_dir):
            return False
    return True


def test_image_eval_scripts_allow_custom_image_dir():
    """
    Any evaluation script using get_image_paths must wire args.image_dir.
    """
    image_eval_files = []

    for script_path in _evaluation_files():
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        uses_image_paths = any(
            _calls_name(node, "get_image_paths") for node in ast.walk(tree)
        )

        if not uses_image_paths:
            continue

        image_eval_files.append(script_path.name)
        assert _has_cli_image_dir_arg(tree), (
            f"{script_path.name} uses get_image_paths " "but has no --image-dir arg"
        )
        assert _has_forwarding_from_args_image_dir(tree), (
            f"{script_path.name} does not pass args.image_dir " "to evaluator calls"
        )
        assert _all_get_image_paths_calls_use_image_dir(tree), (
            f"{script_path.name} calls get_image_paths " "without explicit image_dir"
        )

    assert image_eval_files, "No image-based evaluation scripts were detected"


def _all_get_image_paths_calls_pass_task_category(tree: ast.AST) -> bool:
    """Ensure get_image_paths receives task_category as 3rd positional or keyword arg."""
    calls = [node for node in ast.walk(tree) if _calls_name(node, "get_image_paths")]
    if not calls:
        return False

    for call in calls:
        has_positional = len(call.args) >= 3
        has_keyword = any(
            keyword.arg == "task_category" for keyword in call.keywords
        )
        if not (has_positional or has_keyword):
            return False
    return True


def test_image_eval_scripts_pass_task_category_to_get_image_paths():
    """
    Any evaluation script using get_image_paths must pass task_category so
    that Tier III paired questions receive the correct combined images.
    """
    for script_path in _evaluation_files():
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        uses_image_paths = any(
            _calls_name(node, "get_image_paths") for node in ast.walk(tree)
        )
        if not uses_image_paths:
            continue

        assert _all_get_image_paths_calls_pass_task_category(tree), (
            f"{script_path.name} calls get_image_paths without task_category"
        )


def test_all_eval_scripts_call_shuffle_options():
    """Ensure each evaluator uses shuffle_options before model input."""
    expected_scripts = {
        "eval_anthropic.py",
        "eval_openai.py",
        "eval_openai_text.py",
        "eval_gemini.py",
        "eval_qwen3_vl.py",
        "eval_chatts.py",
    }
    seen_scripts = set()

    for script_path in _evaluation_files():
        if script_path.name not in expected_scripts:
            continue

        seen_scripts.add(script_path.name)
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        has_shuffle_call = any(
            _calls_name(node, "shuffle_options") for node in ast.walk(tree)
        )
        assert has_shuffle_call, (
            f"{script_path.name} must call shuffle_options in evaluation flow"
        )

    assert seen_scripts == expected_scripts


def test_eval_scripts_preserve_full_question_in_model_input():
    """
    Ensure each evaluator includes the full benchmark question in model input.
    """
    benchmark_question = _load_first_benchmark_question()
    assert benchmark_question
    assert "Time-series:" in benchmark_question
    assert "\n" in benchmark_question

    expected_scripts = {
        "eval_anthropic.py",
        "eval_openai.py",
        "eval_openai_text.py",
        "eval_gemini.py",
        "eval_qwen3_vl.py",
        "eval_chatts.py",
    }
    seen_scripts = set()

    for script_path in _evaluation_files():
        if script_path.name not in expected_scripts:
            continue

        seen_scripts.add(script_path.name)
        tree = ast.parse(script_path.read_text(encoding="utf-8"))

        if script_path.name in {
            "eval_anthropic.py",
            "eval_openai.py",
            "eval_openai_text.py",
        }:
            get_response_fn = _function_def(tree, "get_model_response")
            assert get_response_fn is not None
            assert not _function_uses_question_split_on_time_series(get_response_fn), (
                f"{script_path.name} must not truncate question on Time-series"
            )
            assert _function_has_text_payload_with_question(get_response_fn), (
                f"{script_path.name} model payload must include full question text"
            )
            assert _function_has_call_with_keyword(
                get_response_fn, "messages"
            ), f"{script_path.name} must send messages payload to model API"

            process_fn = _function_def(tree, "process_single_question")
            assert process_fn is not None
            assert _function_calls_with_row_question(
                process_fn, "get_model_response"
            ), f"{script_path.name} must pass row['question'] to get_model_response"

        elif script_path.name == "eval_gemini.py":
            get_response_fn = _function_def(tree, "get_model_response")
            assert get_response_fn is not None
            assert not _function_uses_question_split_on_time_series(get_response_fn), (
                f"{script_path.name} must not truncate question on Time-series"
            )
            assert _function_assigns_name_from_question(
                get_response_fn, "user_prompt", "question"
            ), f"{script_path.name} must build user_prompt from full question"
            assert _function_has_call_with_keyword(
                get_response_fn, "contents", "contents"
            ), f"{script_path.name} must send contents payload to model API"

            process_fn = _function_def(tree, "process_single_question")
            assert process_fn is not None
            assert _function_calls_with_row_question(
                process_fn, "get_model_response"
            ), f"{script_path.name} must pass row['question'] to get_model_response"

        elif script_path.name == "eval_qwen3_vl.py":
            prep_fn = _function_def(tree, "prepare_batch_data")
            assert prep_fn is not None
            assert _function_assigns_from_row_question(prep_fn, "question"), (
                "eval_qwen3_vl.py must construct prompt question from row['question']"
            )
            assert _function_calls_with_named_argument(
                prep_fn, "get_model_prompt", "question"
            ), "eval_qwen3_vl.py must pass the full question into get_model_prompt"

            prompt_fn = _function_def(tree, "get_model_prompt")
            assert prompt_fn is not None
            assert _function_has_text_payload_with_question(prompt_fn), (
                "eval_qwen3_vl.py user message content must include question text"
            )

        elif script_path.name == "eval_chatts.py":
            prompt_fn = _function_def(tree, "create_chatts_prompt")
            assert prompt_fn is not None
            assert _function_assigns_name_from_question(
                prompt_fn, "prompt", "question"
            ), "eval_chatts.py prompt must include full question text"

            process_fn = _function_def(tree, "process_single_question")
            assert process_fn is not None
            assert _function_calls_with_row_question(
                process_fn, "get_model_response"
            ), "eval_chatts.py must pass row['question'] to get_model_response"

    assert seen_scripts == expected_scripts


def test_shuffle_options_has_basic_uniformity_on_benchmark_question():
    """
    Repeated shuffles should distribute options across positions roughly evenly.
    """
    question_text, option_values = _load_uniformity_question_and_options()
    assert question_text
    assert len(option_values) >= 2

    source_options = json.dumps(option_values, ensure_ascii=False)
    num_options = len(option_values)
    num_trials = 4000

    seen_orders = set()
    position_counts = {option: [0] * num_options for option in option_values}

    for _ in range(num_trials):
        shuffled_options_text = shuffle_options(source_options)
        shuffled_options = json.loads(shuffled_options_text)

        assert isinstance(shuffled_options, list)
        assert len(shuffled_options) == num_options
        assert sorted(shuffled_options) == sorted(option_values)

        seen_orders.add(tuple(shuffled_options))
        for index, option in enumerate(shuffled_options):
            position_counts[option][index] += 1

    assert len(seen_orders) > 1

    expected_count = num_trials / num_options
    tolerance = expected_count * 0.2
    for option in option_values:
        for count in position_counts[option]:
            assert abs(count - expected_count) <= tolerance


def test_benchmark_every_row_has_expected_option_count():
    """
    Every question must have exactly the expected number of answer choices:
    2 for Anomaly Presence, 5 for all other categories.
    The correct answer must always be among the options.
    """
    benchmark_path = (
        Path(__file__).resolve().parents[1] / "data" / "arfbench-qa.csv"
    )
    expected_counts = {
        "Anomaly Presence": 2,
    }
    default_count = 5

    with benchmark_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_num, row in enumerate(reader):
            category = row["task_category"]
            expected = expected_counts.get(category, default_count)

            options = json.loads(row["options_str"])
            assert len(options) == expected, (
                f"Row {row_num} ({category}): expected {expected} "
                f"options, got {len(options)}"
            )

            correct = row["correct_answer"].strip()
            option_values = [str(o).strip() for o in options]
            assert correct in option_values, (
                f"Row {row_num} ({category}): correct_answer "
                f"{correct!r} not in options"
            )
