#!/usr/bin/env python3
# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache-2.0 License.
#
# This product includes software developed at Datadog
# (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""AST-level regression tests for ablation_gpt experiment routing."""

import ast
from pathlib import Path


ABLATION_GPT_PATH = Path(__file__).resolve().parents[1] / "ablations" / "ablation_gpt.py"


def _parse_ablation_tree() -> ast.AST:
    source = ABLATION_GPT_PATH.read_text(encoding="utf-8")
    return ast.parse(source)


def _function_def(tree: ast.AST, function_name: str) -> ast.AsyncFunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name:
            return node
    return None


def _calls_name(node: ast.AST, name: str) -> bool:
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == name
        ):
            return True
    return False


def _calls_name_in_body(branch: ast.If, name: str) -> bool:
    for statement in branch.body:
        if _calls_name(statement, name):
            return True
    return False


def _is_experiment_eq(node: ast.AST, experiment_name: str) -> bool:
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "experiment_type"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == experiment_name
    )


def _find_experiment_branches(
    process_fn: ast.AsyncFunctionDef, experiment_name: str
) -> list[ast.If]:
    branches = []
    for node in ast.walk(process_fn):
        if isinstance(node, ast.If) and _is_experiment_eq(node.test, experiment_name):
            branches.append(node)
    return branches


def _has_cli_image_dir_arg(tree: ast.AST) -> bool:
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


def test_no_caption_has_dedicated_pipeline():
    tree = _parse_ablation_tree()
    process_fn = _function_def(tree, "process_single_question_ablation")
    assert process_fn is not None

    no_caption_fn = _function_def(tree, "get_model_response_no_caption")
    assert no_caption_fn is not None

    no_caption_branches = _find_experiment_branches(process_fn, "no_caption")
    assert no_caption_branches
    assert any(
        _calls_name_in_body(branch, "get_model_response_no_caption")
        for branch in no_caption_branches
    )
    assert not any(
        _calls_name_in_body(branch, "get_model_response_with_incident")
        for branch in no_caption_branches
    )
    assert not any(
        _calls_name_in_body(branch, "get_model_response_text_only")
        for branch in no_caption_branches
    )


def test_no_caption_pipeline_does_not_accept_incident_context():
    tree = _parse_ablation_tree()
    no_caption_fn = _function_def(tree, "get_model_response_no_caption")
    assert no_caption_fn is not None

    arg_names = [arg.arg for arg in no_caption_fn.args.args]
    assert "incident_context" not in arg_names


def test_ablation_cli_image_dir_is_forwarded_to_evaluator():
    tree = _parse_ablation_tree()
    assert _has_cli_image_dir_arg(tree)
    assert _has_forwarding_from_args_image_dir(tree)
