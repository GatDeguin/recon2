import torch
from infer import beam_search


def test_beam_search_basic():
    logits = torch.tensor([[[0.0, 0.0, 0.0, 5.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 5.0],
                            [0.0, 0.0, 5.0, 0.0, 0.0]]])
    tokens = beam_search(logits, None, beam=2)
    assert tokens == [1, 3, 4, 2]
