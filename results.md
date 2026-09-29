# Results

One row per change, with the exact command that produced it. This table is the ablation appendix.

## Metric choice

Each mention carries exactly one gold CUI (`featurization/preprocessor.py:113` → one-hot `Y` rows),
so this is extreme multi-**class**, not multi-label. Reported metrics are **acc@1, MRR, recall@k**.
PECOS's precision@k / propensity-scored suite is **not** reused: with a single gold label,
precision@k is just recall@k / k and carries no extra information.

`recall@candidates` is the old binary `gold in cand` hit count, kept so earlier numbers stay
comparable. `In-vocabulary mentions` is the ceiling from defect #10 — mentions whose gold CUI is
absent from the training label space are deleted before scoring
(`test/xmr4el/test_evaluate_pipeline.py`), so every row below is an in-vocabulary upper bound.

## Runs

| # | Change | Dataset / cmd | in-vocab | acc@1 | MRR | R@5 | R@20 | R@cand |
|---|--------|---------------|----------|-------|-----|-----|------|--------|
| 0 | Phase 0 baseline (metrics added, `topk_mode` back to `per_leaf`) | st21pv dev, `-ds_len 2000` | — | — | — | — | — | — |

Row 0 is not filled in: no runnable Python environment on this machine (see Blockers in the session
notes — `.venv` points at `/usr/bin/python3`, a Command Line Tools stub).

## Commands

```bash
python test/xmr4el/test_train_pipeline.py \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -model_config .models/xmr4el_base_config.json -ds_len 2000

python test/xmr4el/test_evaluate_pipeline.py \
  -xmodel_path test/test_data/saved_trees/<run> \
  -test_path datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt \
  -beam_size 5 -topk 20

python test/xmr4el/test_evaluate_pipeline.py -selfcheck   # asserts on gold_rank
```
