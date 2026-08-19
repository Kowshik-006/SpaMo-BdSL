import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np, glob
from transformers import AutoTokenizer

LIMIT = 64
tok = AutoTokenizer.from_pretrained("google/flan-t5-xl", cache_dir="./hf_cache", use_fast=False)

def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

for f in sorted(glob.glob("preprocess/Phoenix14T/*_info_ml.npy")):
    d = np.load(f, allow_pickle=True).item()
    lengths = []
    over = []
    for i in d:
        if i == "prefix":
            continue
        text = d[i]["text"]
        n = len(tok(text).input_ids)  # includes appended </s>, matches training tokenization
        lengths.append(n)
        if n > LIMIT:
            over.append((n, text))
    lengths_sorted = sorted(lengths)
    name = os.path.basename(f)
    print("=" * 70)
    print(f"{name}  (n={len(lengths)})")
    print(f"  min={min(lengths)}  mean={sum(lengths)/len(lengths):.1f}  "
          f"median={pct(lengths_sorted,50):.0f}  p90={pct(lengths_sorted,90):.0f}  "
          f"p95={pct(lengths_sorted,95):.0f}  p99={pct(lengths_sorted,99):.0f}  max={max(lengths)}")
    ge60 = sum(1 for x in lengths if x >= 60)
    gt64 = sum(1 for x in lengths if x > 64)
    print(f"  >64 tokens (truncated): {gt64} ({100*gt64/len(lengths):.2f}%) | "
          f">=60 tokens (borderline): {ge60} ({100*ge60/len(lengths):.2f}%)")
    # for n, text in sorted(over, reverse=True)[:10]:
    #     print(f"    [{n}] {text}")
