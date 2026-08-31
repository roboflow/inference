"""Tests for the fine-tune span format: its grammar and its parser."""

import pytest
import torch

from inference_models.models.base.action_recognition import ActionRecognitionPrediction
from inference_models.models.cosmos3.span_format import (
    SpanConstrainedDecoder,
    parse_spans,
)


class _FakeTokenizer:
    def __init__(self, class_names=()):
        ordinary_tokens = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            ".",
            "<",
            ">",
            " ",
            "\n",
            "n",
            "o",
            "e",
            "none",
            "x",
        ]
        self.unk_token_id = 0
        self.eos_token_id = 99
        self.pad_token_id = 99
        self.eos_token = "<|im_end|>"
        self._vocabulary = {
            token: index + 1 for index, token in enumerate(ordinary_tokens)
        }
        self._added_vocabulary = {
            f"<|cls:{class_name}|>": 50 + index
            for index, class_name in enumerate(class_names)
        }
        self._vocabulary.update(self._added_vocabulary)
        self._id_to_token = {
            token_id: token for token, token_id in self._vocabulary.items()
        }

    def get_added_vocab(self):
        return dict(self._added_vocabulary)

    def get_vocab(self):
        return dict(self._vocabulary)

    def convert_tokens_to_ids(self, token):
        return self._vocabulary.get(token, self.unk_token_id)

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(self._id_to_token.get(token_id, "") for token_id in token_ids)

    def id(self, token):
        return self._vocabulary[token]


def _prediction(start, end, label):
    return ActionRecognitionPrediction(
        start_frame_idx=start,
        end_frame_idx=end,
        class_name=label,
    )


def test_constrained_decoder_allowed_tokens_follow_line_grammar() -> None:
    tokenizer = _FakeTokenizer(class_names=["walking"])
    class_token_id = tokenizer.id("<|cls:walking|>")
    grammar = SpanConstrainedDecoder(
        tokenizer=tokenizer,
        class_token_ids={"walking": class_token_id},
    )

    start_allowed = grammar(0, torch.tensor([], dtype=torch.int64))
    assert class_token_id in start_allowed
    assert tokenizer.id("n") in start_allowed
    assert tokenizer.id("none") in start_allowed
    assert tokenizer.eos_token_id not in start_allowed

    prefix_through_open_angle = torch.tensor(
        [class_token_id, tokenizer.id(" "), tokenizer.id("<")]
    )
    after_open_angle = grammar(0, prefix_through_open_angle)
    assert tokenizer.id("0") in after_open_angle
    assert tokenizer.id("9") in after_open_angle
    assert tokenizer.id(".") not in after_open_angle
    assert tokenizer.eos_token_id not in after_open_angle

    complete_line = torch.tensor(
        [
            class_token_id,
            tokenizer.id(" "),
            tokenizer.id("<"),
            tokenizer.id("1"),
            tokenizer.id("."),
            tokenizer.id("2"),
            tokenizer.id(">"),
            tokenizer.id(" "),
            tokenizer.id("<"),
            tokenizer.id("3"),
            tokenizer.id("."),
            tokenizer.id("4"),
            tokenizer.id(">"),
        ]
    )
    line_boundary_allowed = grammar(0, complete_line)
    assert tokenizer.eos_token_id in line_boundary_allowed
    assert tokenizer.id("\n") in line_boundary_allowed
    assert class_token_id not in line_boundary_allowed

    after_newline = torch.cat([complete_line, torch.tensor([tokenizer.id("\n")])])
    # A completed line may end the answer, so EOS is allowed beside the
    # next class token.
    assert grammar(0, after_newline) == sorted([class_token_id, tokenizer.eos_token_id])


def test_fine_tune_parser_accepts_valid_lines_and_converts_seconds() -> None:
    result = parse_spans(
        text=("<|cls:walking|> <-0.20> <0.21>\n" "<|cls:running|> <1.40> <0.20>"),
        class_names=["walking", "running"],
        num_frames=10,
        fps=5.0,
    )

    assert result == [
        _prediction(0, 2, "walking"),
        _prediction(1, 7, "running"),
    ]


def test_fine_tune_parser_drops_out_of_vocabulary_labels() -> None:
    result = parse_spans(
        text=("<|cls:walking|> <0.00> <0.20>\n" "<|cls:jumping|> <0.20> <0.40>"),
        class_names=["walking"],
        num_frames=5,
        fps=5.0,
    )

    assert result == [_prediction(0, 1, "walking")]


@pytest.mark.parametrize("answer", ["none", "  NONE  ", "not parseable"])
def test_fine_tune_parser_returns_empty_for_none_or_no_valid_lines(answer) -> None:
    assert (
        parse_spans(
            text=answer,
            class_names=["walking"],
            num_frames=10,
            fps=5.0,
        )
        == []
    )
