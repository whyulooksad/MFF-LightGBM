import torch

from developer.representation.dataset import FlowDataset
from user_app.inference.semantic_encoder import FeatureDataset


class CountingTokenizer:
    def __init__(self):
        self.calls = 0

    def __call__(self, _text, **_kwargs):
        self.calls += 1
        return {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }


def test_lora_dataset_tokenizes_lazily():
    tokenizer = CountingTokenizer()
    dataset = FlowDataset([{"text": "flow", "label": 2}], tokenizer)
    assert tokenizer.calls == 0
    item = dataset[0]
    assert tokenizer.calls == 1
    assert item["labels"].item() == 2


def test_feature_dataset_tokenizes_lazily_and_allows_missing_label():
    tokenizer = CountingTokenizer()
    dataset = FeatureDataset([{"text": "flow", "label": None}], tokenizer)
    assert tokenizer.calls == 0
    item = dataset[0]
    assert tokenizer.calls == 1
    assert item["labels"].item() == -1
