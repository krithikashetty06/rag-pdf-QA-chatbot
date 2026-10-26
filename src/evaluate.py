"""Evaluate retrieval and answer quality, and run chunking/retrieval ablations.

Examples (run from the project root):
    python -m src.evaluate                      # retrieval metrics only (free, fast)
    python -m src.evaluate --judge              # + LLM-judged faithfulness/correctness
    python -m src.evaluate --no-hybrid --k 6    # try other settings
    python -m src.evaluate --ablate             # chunk size x retrieval mode table
"""
import argparse
import json
import os
import re

from . import config
from .generate import answer, complete, format_context
from .ingest import build_index, load_pdfs
from .retrieve import Retriever

JUDGE_PROMPT = """You are grading a question-answering system.

Question: {question}
Reference answer: {reference}

Retrieved context:
{context}

System answer: {answer}

Score two things:
- faithful: 1 if every claim in the system answer is supported by the retrieved context, else 0.
- correct: 1 if the system answer matches the reference answer in substance, else 0.

Return ONLY JSON like {{"faithful": 1, "correct": 0}}"""


def load_eval(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def retrieval_metrics(retriever, items, k):
    """Hit rate@k and MRR at page level, over answerable questions."""
    answerable = [i for i in items if i.get("answerable", True)]
    hits, rr = 0, 0.0
    for it in answerable:
        res = retriever.search(it["question"], k)
        keys = [(r["meta"]["source"], r["meta"]["page"]) for r in res]
        target = (it["source"], it["page"])
        if target in keys:
            hits += 1
            rr += 1 / (keys.index(target) + 1)
    n = len(answerable) or 1
    return hits / n, rr / n


def judge(question, reference, chunks, ans):
    raw = complete(JUDGE_PROMPT.format(question=question, reference=reference,
                                       context=format_context(chunks), answer=ans),
                   max_tokens=60)
    m = re.search(r"\{.*?\}", raw, re.S)
    try:
        d = json.loads(m.group(0))
        return int(d.get("faithful", 0)), int(d.get("correct", 0))
    except Exception:
        return 0, 0


def answer_metrics(retriever, items, k):
    faith = correct = n_ans = 0
    abstained = n_unans = 0
    for it in items:
        chunks = retriever.search(it["question"], k)
        ans = answer(it["question"], chunks)
        if it.get("answerable", True):
            f, c = judge(it["question"], it["answer"], chunks, ans)
            faith += f
            correct += c
            n_ans += 1
        else:
            n_unans += 1
            abstained += int("couldn't find" in ans.lower())
    return {
        "faithfulness": faith / (n_ans or 1),
        "correctness": correct / (n_ans or 1),
        "abstention_rate": abstained / (n_unans or 1),
    }


def run_ablation(items, k, with_rerank):
    pages = load_pdfs()
    modes = [("vector", False, False), ("hybrid", True, False)]
    if with_rerank:
        modes.append(("hybrid+rerank", True, True))
    rows = []
    for size, overlap in [(400, 50), (800, 150), (1200, 200)]:
        name = f"abl_{size}_{overlap}"
        print(f"Building index {name} ...")
        build_index(pages, collection=name, chunk_size=size, overlap=overlap)
        for label, hyb, rer in modes:
            r = Retriever(collection=name, hybrid=hyb, rerank=rer)
            hit, mrr = retrieval_metrics(r, items, k)
            rows.append((size, overlap, label, hit, mrr))
            print(f"  {label:14s} hit@{k}={hit:.2f} mrr={mrr:.2f}")
    table = [f"| Chunk size | Overlap | Mode | Hit rate@{k} | MRR |", "|---|---|---|---|---|"]
    table += [f"| {s} | {o} | {m} | {h:.2f} | {mr:.2f} |" for s, o, m, h, mr in rows]
    os.makedirs("results", exist_ok=True)
    with open("results/ablation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(table))
    print("\n" + "\n".join(table))
    print("\nSaved to results/ablation.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval_questions.json")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--collection", default=config.COLLECTION)
    ap.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=config.USE_HYBRID)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--judge", action="store_true", help="LLM-judge answer quality (uses API calls)")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--with-rerank", action="store_true", help="include reranker in --ablate")
    args = ap.parse_args()

    items = load_eval(args.questions)
    if args.ablate:
        run_ablation(items, args.k, args.with_rerank)
        raise SystemExit

    retriever = Retriever(collection=args.collection, hybrid=args.hybrid, rerank=args.rerank)
    hit, mrr = retrieval_metrics(retriever, items, args.k)
    print(f"Hit rate@{args.k}: {hit:.3f}\nMRR: {mrr:.3f}")
    if args.judge:
        for name, val in answer_metrics(retriever, items, args.k).items():
            print(f"{name}: {val:.3f}")
