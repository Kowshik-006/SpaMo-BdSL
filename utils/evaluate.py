from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF, TER


def _lcs_length(x, y):
    """Compute length of the longest common subsequence between two sequences."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _rouge_l(reference, prediction):
    """Compute ROUGE-L precision, recall, and F1 using whitespace tokenization."""
    ref_tokens = reference.split()
    pred_tokens = prediction.split()

    if len(ref_tokens) == 0 and len(pred_tokens) == 0:
        return 1.0, 1.0, 1.0
    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0, 0.0, 0.0

    lcs = _lcs_length(ref_tokens, pred_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_results(predictions, references, split="train", device='cpu', tokenizer='13a', use_stemmer=True):
    """
    Evaluate prediction results using BLEU and ROUGE metrics.

    Args:
        predictions (list): List of predicted sequences.
        references (list): List of reference sequences.
        tokenizer (object, optional): Tokenizer if needed for evaluation.
        split (str): The data split being evaluated.
        use_stemmer (bool): Whether to use English stemmer for ROUGE.
            Set to False for non-Latin scripts (e.g. Bangla) to use
            a Unicode-safe whitespace-based ROUGE-L instead.

    Returns:
        dict: A dictionary of evaluation scores.
    """
    log_dicts = {}

    bleu4 = BLEU(max_ngram_order=4, tokenize=tokenizer).corpus_score(predictions, [references]).score
    log_dicts[f"{split}/bleu4"] = bleu4

    if split == 'test':
        for i in range(1, 4):
            score = BLEU(max_ngram_order=i, tokenize=tokenizer).corpus_score(predictions, [references]).score
            log_dicts[f"{split}/bleu" + str(i)] = score

        if use_stemmer:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_scores = [scorer.score(ref, pred)['rougeL'] for ref, pred in zip(references, predictions)]
            avg_precision = sum(s.precision for s in rouge_scores) / len(rouge_scores)
            avg_recall = sum(s.recall for s in rouge_scores) / len(rouge_scores)
            avg_f1 = sum(s.fmeasure for s in rouge_scores) / len(rouge_scores)
        else:
            scores = [_rouge_l(ref, pred) for ref, pred in zip(references, predictions)]
            avg_precision = sum(s[0] for s in scores) / len(scores)
            avg_recall = sum(s[1] for s in scores) / len(scores)
            avg_f1 = sum(s[2] for s in scores) / len(scores)

        log_dicts[f"{split}/rougeL_precision"] = avg_precision
        log_dicts[f"{split}/rougeL_recall"] = avg_recall
        log_dicts[f"{split}/rougeL_f1"] = avg_f1

    return log_dicts