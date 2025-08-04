import csv
from pathlib import Path


def _levenshtein(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def _rates(refs, hyps):
    word_errs = char_errs = total_words = total_chars = 0
    for ref, hyp in zip(refs, hyps):
        r_words, h_words = ref.split(), hyp.split()
        word_errs += _levenshtein(r_words, h_words)
        total_words += len(r_words)
        r_chars = list(ref.replace(" ", ""))
        h_chars = list(hyp.replace(" ", ""))
        char_errs += _levenshtein(r_chars, h_chars)
        total_chars += len(r_chars)
    wer = word_errs / total_words if total_words else 0.0
    cer = char_errs / total_chars if total_chars else 0.0
    return wer, cer


def test_public_subset_metrics():
    data_file = Path(__file__).parent / "data" / "librispeech_subset.csv"
    refs, a_preds, b_preds = [], [], []
    with data_file.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            refs.append(row["reference"])
            a_preds.append(row["model_a"])
            b_preds.append(row["model_b"])
    wer_a, cer_a = _rates(refs, a_preds)
    wer_b, cer_b = _rates(refs, b_preds)
    assert abs(wer_a - 0.15) < 1e-6
    assert abs(cer_a - 0.11688311688311688) < 1e-6
    assert wer_b == 0.0 and cer_b == 0.0
    assert wer_a > wer_b
