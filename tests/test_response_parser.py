import pytest

from superagent.response_parser import parse_json_content
from superagent.usage import InvalidPlanError


@pytest.mark.parametrize("prefix", ["I should return the requested object.", "<think>I should return the requested object."])
def test_qwen_reasoning_prefix_with_optional_opening_tag(prefix):
    content = prefix + '\n</think>\n\n{"ok":true}'
    assert parse_json_content(content, "qwen_think_json") == {"ok": True}


def test_default_parser_rejects_qwen_reasoning_prefix():
    with pytest.raises(InvalidPlanError):
        parse_json_content('Reasoning\n</think>\n{"ok":true}')


@pytest.mark.parametrize("mode", ["json", "qwen_think_json"])
def test_raw_json_remains_valid_in_both_modes(mode):
    assert parse_json_content(' {"ok":true,"text":"</think>"}\n', mode) == {"ok": True, "text": "</think>"}


def test_json_in_reasoning_is_never_selected_as_answer():
    content = 'A possible answer was {"ok":false}, but that was wrong.\n</think>\n{"ok":true}'
    assert parse_json_content(content, "qwen_think_json") == {"ok": True}


@pytest.mark.parametrize("content", [
    'Reasoning</think>more reasoning</think>{"ok":true}',
    'Reasoning</think>{"ok":true} trailing explanation',
    'Reasoning</think>```json\n{"ok":true}\n```',
    'Reasoning with {"ok":true} but no closing marker',
    'Reasoning</think>',
    'Reasoning</think>{invalid JSON}',
])
def test_qwen_mode_rejects_ambiguous_or_incomplete_answers(content):
    with pytest.raises(InvalidPlanError):
        parse_json_content(content, "qwen_think_json")


@pytest.mark.parametrize("content", [None, {}, [], 1, True, b'{"ok":true}'])
def test_non_string_content_is_rejected(content):
    with pytest.raises(InvalidPlanError):
        parse_json_content(content, "qwen_think_json")


@pytest.mark.parametrize("mode", [None, "unknown", [], {}])
def test_unsupported_parser_mode_is_rejected(mode):
    with pytest.raises(InvalidPlanError):
        parse_json_content('{"ok":true}', mode)


@pytest.mark.parametrize("content", ['{"ok":true} trailing', '```json\n{"ok":true}\n```', '{invalid}', ''])
def test_default_mode_has_no_json_extraction_or_markdown_repair(content):
    with pytest.raises(InvalidPlanError):
        parse_json_content(content)
