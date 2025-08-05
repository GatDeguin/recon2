from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from train_lm import build_vocab, TextDataset, train_model  # type: ignore
from models.transformer_lm import TransformerLanguageModel, load_model


def test_train_and_load_lm(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hola mundo\n")

    vocab_path = tmp_path / "vocab.txt"
    vocab = build_vocab(corpus, vocab_path)

    ds = TextDataset(corpus, vocab, seq_len=3)
    loader = DataLoader(ds, batch_size=1)

    model = TransformerLanguageModel(len(vocab))
    train_model(model, loader, epochs=1, lr=1e-2, pad_id=vocab["<pad>"], device=torch.device("cpu"))

    ckpt = tmp_path / "lm.pt"
    torch.save(model.state_dict(), ckpt)

    lm = load_model(str(ckpt), vocab_size=len(vocab))
    x = torch.tensor([[vocab["<sos>"], vocab["hola"], vocab["mundo"]]])
    with torch.no_grad():
        out = lm(x)
    assert out.shape == (1, 3, len(vocab))
