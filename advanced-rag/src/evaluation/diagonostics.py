"""
Diagnostics that explain WHY the reranker helps or hurts.

Each per-query record (built in evaluate_retrievers.py) contains:
    query_id, query, relevant (set of doc ids)
    pool_ids        candidate pool in hybrid order (best first)
    hybrid_scores   {doc_id: RRF score}
    rerank_scores   {doc_id: reranker score}   (empty if no reranker)
    rerank_order    pool re-sorted by reranker score
    bm25_ids, dense_ids   top-k lists from each standalone retriever
    latency         {"bm25", "dense", "hybrid", "rerank"} in seconds

Outputs (in out_dir):
    diagnostics_report.txt     everything printed to the console
    diagnostics.png            helped-vs-hurt, alpha sweep, rank histogram
    alpha_sweep.csv            metrics for reranker/hybrid blend weights 0..1
    per_query_diagnostics.csv  rank of first relevant doc per method, per query
    hurt_cases.txt             queries where reranking lost a hit hybrid had
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import matplotlib.pyplot as plt

from .metrics import aggregate, evaluate_single_query

K_LIST = (1, 3, 5, 10)
LONG_DOC_WORDS = 380  # ~512 tokens for English text: beyond this the reranker truncates


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

class _Report:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def add(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines), encoding="utf-8")


def _first_rank(ranked: Sequence[str], relevant: Set[str]) -> Optional[int]:
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return rank
    return None


def _minmax(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in values.items()}


def fused_order(pool_ids, hybrid_scores, rerank_scores, alpha: float) -> List[str]:
    rr, hy = _minmax(rerank_scores), _minmax(hybrid_scores)
    fused = {c: alpha * rr.get(c, 0.0) + (1 - alpha) * hy.get(c, 0.0) for c in pool_ids}
    return sorted(pool_ids, key=lambda c: fused[c], reverse=True)


def _spearman(order_a: Sequence[str], order_b: Sequence[str]) -> Optional[float]:
    n = len(order_a)
    if n < 3:
        return None
    pos_b = {doc: i for i, doc in enumerate(order_b)}
    d2 = sum((i - pos_b[doc]) ** 2 for i, doc in enumerate(order_a))
    return 1 - 6 * d2 / (n * (n * n - 1))


def _snippet(text: str, n: int = 220) -> str:
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# ----------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------

def run_diagnostics(
    records: List[dict],
    out_dir: Path,
    text_by_id: Dict[str, str],
    top_k: int,
    candidate_pool: int,
    fusion_alpha: float,
    reranker_name: str,
    reranker_max_length: int,
) -> None:
    rep = _Report()
    n = len(records)
    has_rr = any(r["rerank_scores"] for r in records)

    rep.add("\n" + "=" * 70)
    rep.add(f"DIAGNOSTICS  ({n} queries | pool={candidate_pool} | top_k={top_k})")
    rep.add(f"Reranker: {reranker_name if has_rr else 'NOT RUN'}")
    rep.add("=" * 70)

    # ---- 1. Candidate-pool ceiling -------------------------------------
    rep.add("\n[1] CANDIDATE POOL CEILING  (can the reranker even succeed?)")
    pool_rate = _mean(
        1.0 if any(d in r["relevant"] for d in r["pool_ids"]) else 0.0 for r in records
    )
    rep.add(f"  Queries with a relevant doc somewhere in the pool: {_pct(pool_rate)}")
    rep.add("  A reranker only reorders the pool, so this is its maximum possible hit rate.")
    if has_rr:
        rep.add(f"  {'k':<4}{'hybrid':>10}{'reranked':>11}{'ceiling':>10}{'headroom':>11}")
        for k in K_LIST:
            hy = _mean(
                1.0 if (_first_rank(r["pool_ids"], r["relevant"]) or 10**9) <= k else 0.0
                for r in records
            )
            rr = _mean(
                1.0 if (_first_rank(r["rerank_order"], r["relevant"]) or 10**9) <= k else 0.0
                for r in records
            )
            rep.add(f"  {k:<4}{_pct(hy):>10}{_pct(rr):>11}{_pct(pool_rate):>10}{_pct(pool_rate - rr):>11}")

    # ---- 2. Helped vs hurt --------------------------------------------
    helped_hurt = {}
    if has_rr:
        rep.add("\n[2] HELPED vs HURT  (hybrid alone -> hybrid + reranker, per query)")
        rep.add(f"  {'k':<4}{'helped':>8}{'hurt':>7}{'both hit':>10}{'both miss':>11}{'net':>6}")
        for k in K_LIST:
            helped = hurt = both = neither = 0
            for r in records:
                before = (_first_rank(r["pool_ids"], r["relevant"]) or 10**9) <= k
                after = (_first_rank(r["rerank_order"], r["relevant"]) or 10**9) <= k
                helped += (not before) and after
                hurt += before and (not after)
                both += before and after
                neither += (not before) and (not after)
            helped_hurt[k] = (helped, hurt)
            rep.add(f"  {k:<4}{helped:>8}{hurt:>7}{both:>10}{neither:>11}{helped - hurt:>+6}")

        # rank movement of the first relevant doc
        moved_up = moved_down = same = to_top = from_top = 0
        ranks_h, ranks_r = [], []
        for r in records:
            rh = _first_rank(r["pool_ids"], r["relevant"])
            rr_ = _first_rank(r["rerank_order"], r["relevant"])
            if rh is None or rr_ is None:
                continue
            ranks_h.append(rh)
            ranks_r.append(rr_)
            moved_up += rr_ < rh
            moved_down += rr_ > rh
            same += rr_ == rh
            to_top += rh > 1 and rr_ == 1
            from_top += rh == 1 and rr_ > 1
        rep.add("\n  Rank of the first relevant doc (queries where it is in the pool):")
        rep.add(f"    moved up: {moved_up} | moved down: {moved_down} | unchanged: {same}")
        rep.add(f"    promoted to #1: {to_top} | demoted from #1: {from_top}")
        if ranks_h:
            rep.add(
                f"    mean rank  hybrid={_mean(ranks_h):.2f}  reranked={_mean(ranks_r):.2f}"
                f" | median  hybrid={statistics.median(ranks_h)}  reranked={statistics.median(ranks_r)}"
            )

        # ---- 3. Score separation --------------------------------------
        rep.add("\n[3] RERANKER SCORE SEPARATION  (does it score relevant docs higher?)")
        rel_norm, non_norm, rel_raw, non_raw, spearmans = [], [], [], [], []
        for r in records:
            norm = _minmax(r["rerank_scores"])
            for doc_id, raw in r["rerank_scores"].items():
                if doc_id in r["relevant"]:
                    rel_norm.append(norm[doc_id])
                    rel_raw.append(raw)
                else:
                    non_norm.append(norm[doc_id])
                    non_raw.append(raw)
            rho = _spearman(r["pool_ids"], r["rerank_order"])
            if rho is not None:
                spearmans.append(rho)
        sep = _mean(rel_norm) - _mean(non_norm)
        rep.add(f"  Mean raw score      relevant={_mean(rel_raw):.4f} | non-relevant={_mean(non_raw):.4f}")
        rep.add(f"  Mean normalized     relevant={_mean(rel_norm):.3f} | non-relevant={_mean(non_norm):.3f}"
                f" | separation={sep:.3f}")
        rep.add(f"  Agreement with hybrid order (Spearman, 1=identical): {_mean(spearmans):.3f}")
        rep.add("  (Low separation = reranker barely distinguishes right from wrong docs;")
        rep.add("   low agreement = it strongly disagrees with the retrievers.)")
    else:
        sep = None
        spearmans = []

    # ---- 4. Retriever overlap -----------------------------------------
    rep.add("\n[4] RETRIEVER OVERLAP  (why is hybrid ~ semantic?)")
    jac_k, jac_5, same5, same_k = [], [], 0, 0
    for r in records:
        b, d = set(r["bm25_ids"]), set(r["dense_ids"])
        if b | d:
            jac_k.append(len(b & d) / len(b | d))
        b5, d5 = set(r["bm25_ids"][:5]), set(r["dense_ids"][:5])
        if b5 | d5:
            jac_5.append(len(b5 & d5) / len(b5 | d5))
        h5 = set(r["pool_ids"][:5])
        same5 += h5 == d5
        same_k += set(r["pool_ids"][:top_k]) == set(r["dense_ids"][:top_k])
    rep.add(f"  BM25 vs semantic top-{top_k} Jaccard: {_mean(jac_k):.3f} | top-5: {_mean(jac_5):.3f}")
    rep.add(f"  Hybrid top-5 identical to semantic top-5 (as a set): {_pct(same5 / n)} of queries")
    rep.add(f"  Hybrid top-{top_k} identical to semantic top-{top_k} (as a set): {_pct(same_k / n)} of queries")

    # ---- 5. Text-length stats -----------------------------------------
    rep.add("\n[5] TEXT LENGTH  (truncation risk for the reranker)")
    q_words = [len(r["query"].split()) for r in records]
    all_docs = [len(t.split()) for t in text_by_id.values()]
    rel_ids = {d for r in records for d in r["relevant"] if d in text_by_id}
    rel_docs = [len(text_by_id[d].split()) for d in rel_ids]
    rep.add(f"  Avg query length: {_mean(q_words):.1f} words")
    rep.add(f"  Avg doc length:   {_mean(all_docs):.1f} words (max {max(all_docs) if all_docs else 0})")
    long_all = _mean(1.0 if w > LONG_DOC_WORDS else 0.0 for w in all_docs)
    long_rel = _mean(1.0 if w > LONG_DOC_WORDS else 0.0 for w in rel_docs)
    rep.add(f"  Docs longer than ~{LONG_DOC_WORDS} words (>{reranker_max_length} tokens): "
            f"{_pct(long_all)} of corpus | {_pct(long_rel)} of relevant docs")

    # ---- 6. Latency ---------------------------------------------------
    rep.add("\n[6] LATENCY  (mean ms per query)")
    for stage in ("bm25", "dense", "hybrid", "rerank"):
        vals = [r["latency"].get(stage) for r in records if r["latency"].get(stage) is not None]
        if vals:
            rep.add(f"  {stage:<8}{1000 * _mean(vals):>10.1f} ms")

    # ---- 7. Alpha sweep -----------------------------------------------
    sweep_rows = []
    best = None
    if has_rr:
        rep.add("\n[7] BLEND SWEEP  (alpha = weight on reranker score; 0 = hybrid, 1 = reranker only)")
        rep.add(f"  {'alpha':<7}{'hit@1':>8}{'hit@5':>8}{'hit@10':>8}{'mrr':>8}{'ndcg@10':>9}")
        for step in range(11):
            alpha = step / 10
            per_query = []
            for r in records:
                order = fused_order(r["pool_ids"], r["hybrid_scores"], r["rerank_scores"], alpha)[:top_k]
                per_query.append(evaluate_single_query(order, r["relevant"], k_values=K_LIST))
            agg = aggregate(per_query)
            row = {"alpha": alpha, **{m: agg[m] for m in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr", "ndcg@10")}}
            sweep_rows.append(row)
            rep.add(f"  {alpha:<7.1f}{_pct(row['hit@1']):>8}{_pct(row['hit@5']):>8}"
                    f"{_pct(row['hit@10']):>8}{row['mrr']:>8.3f}{row['ndcg@10']:>9.3f}")
        best = max(sweep_rows, key=lambda x: x["ndcg@10"])
        rep.add(f"  Best alpha by NDCG@10: {best['alpha']:.1f}  (you ran the main comparison at {fusion_alpha})")
        rep.add("  Caution: picking alpha on the same queries you score on is slightly optimistic;")
        rep.add("  say so in your report, or tune on a subset of queries and report on the rest.")

        with (out_dir / "alpha_sweep.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            w.writeheader()
            w.writerows(sweep_rows)

    # ---- 8. Per-query dump + hurt cases -------------------------------
    with (out_dir / "per_query_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query", "n_relevant", "in_pool",
                    "rank_bm25", "rank_semantic", "rank_hybrid", "rank_reranked", "rank_fused"])
        for r in records:
            fused = (fused_order(r["pool_ids"], r["hybrid_scores"], r["rerank_scores"], fusion_alpha)
                     if has_rr else [])
            w.writerow([
                r["query_id"], r["query"], len(r["relevant"]),
                int(any(d in r["relevant"] for d in r["pool_ids"])),
                _first_rank(r["bm25_ids"], r["relevant"]) or "",
                _first_rank(r["dense_ids"], r["relevant"]) or "",
                _first_rank(r["pool_ids"], r["relevant"]) or "",
                (_first_rank(r["rerank_order"], r["relevant"]) or "") if has_rr else "",
                (_first_rank(fused, r["relevant"]) or "") if has_rr else "",
            ])

    if has_rr:
        hurt = [r for r in records
                if (_first_rank(r["pool_ids"], r["relevant"]) or 10**9) <= 5
                and (_first_rank(r["rerank_order"], r["relevant"]) or 10**9) > 5]
        with (out_dir / "hurt_cases.txt").open("w", encoding="utf-8") as f:
            f.write(f"{len(hurt)} queries where hybrid had a relevant doc in its top 5 "
                    f"but the reranker pushed it out.\nShowing up to 25.\n")
            for r in hurt[:25]:
                rel_in_pool = [d for d in r["pool_ids"] if d in r["relevant"]]
                target = rel_in_pool[0]
                f.write("\n" + "-" * 78 + "\n")
                f.write(f"QUERY: {r['query']}\n")
                f.write(f"Relevant doc {target}: hybrid rank {r['pool_ids'].index(target) + 1} -> "
                        f"reranked rank {r['rerank_order'].index(target) + 1} "
                        f"(rerank score {r['rerank_scores'][target]:.4f})\n")
                f.write(f"  TEXT: {_snippet(text_by_id.get(target, ''))}\n")
                f.write("  Reranker's top 3 instead:\n")
                for doc_id in r["rerank_order"][:3]:
                    tag = "RELEVANT" if doc_id in r["relevant"] else "wrong"
                    f.write(f"    [{tag}] {doc_id} score={r['rerank_scores'][doc_id]:.4f}: "
                            f"{_snippet(text_by_id.get(doc_id, ''), 160)}\n")
        rep.add(f"\n[8] Saved {len(hurt)} 'hurt' cases -> hurt_cases.txt "
                "(read these: they show what the reranker preferred instead)")

    # ---- 9. Automated interpretation ----------------------------------
    rep.add("\n[9] WHAT THIS SUGGESTS")
    if has_rr:
        rr5 = _mean(1.0 if (_first_rank(r["rerank_order"], r["relevant"]) or 10**9) <= 5 else 0.0 for r in records)
        if pool_rate - rr5 >= 0.05:
            rep.add(f"  - Headroom exists: the pool contains a relevant doc for {_pct(pool_rate)} of queries "
                    f"but reranked hit@5 is {_pct(rr5)}. A better reranker could gain.")
        else:
            rep.add("  - The pool is already almost fully exploited; gains must come from a bigger pool.")
        helped5, hurt5 = helped_hurt.get(5, (0, 0))
        if hurt5 > helped5:
            rep.add(f"  - At k=5 the reranker hurt {hurt5} queries and helped {helped5}: net negative.")
        else:
            rep.add(f"  - At k=5 the reranker helped {helped5} queries and hurt {hurt5}: net non-negative.")
        if sep is not None and sep < 0.30:
            rep.add(f"  - Score separation is small ({sep:.2f}): the model is weak at telling relevant "
                    "from irrelevant docs in this domain (typical of an out-of-domain reranker).")
        if spearmans and _mean(spearmans) < 0.3:
            rep.add(f"  - Low agreement with hybrid order ({_mean(spearmans):.2f}): it overrides the "
                    "retrievers heavily, so its mistakes are not corrected by them. Blending helps here.")
        if long_rel > 0.15:
            rep.add(f"  - {_pct(long_rel)} of relevant docs exceed the reranker's length limit and get truncated.")
        if best and 0 < best["alpha"] < 1:
            rep.add(f"  - Blending beats both extremes (best alpha {best['alpha']:.1f}): "
                    "each signal fixes the other's mistakes.")
        elif best and best["alpha"] == 0:
            rep.add("  - The blend never beats hybrid alone: this reranker adds no useful signal on this data.")
    if same5 / n > 0.6:
        rep.add(f"  - Hybrid top-5 equals semantic top-5 for {_pct(same5 / n)} of queries: "
                "the dense retriever dominates the fusion, which is why hybrid ~ semantic.")
    if _mean(jac_k) < 0.3:
        rep.add(f"  - BM25 and semantic overlap little (Jaccard {_mean(jac_k):.2f}): they find different docs, "
                "so a smarter fusion (e.g. weighted RRF) has room to help.")

    # ---- plots ---------------------------------------------------------
    _plot(records, helped_hurt, sweep_rows, fusion_alpha, out_dir, has_rr)
    rep.save(out_dir / "diagnostics_report.txt")
    rep.add(f"\nSaved diagnostics -> {out_dir}")


def _plot(records, helped_hurt, sweep_rows, fusion_alpha, out_dir: Path, has_rr: bool) -> None:
    if not has_rr:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    # panel 1: helped vs hurt
    ax = axes[0]
    ks = list(helped_hurt.keys())
    xs = range(len(ks))
    ax.bar([x - 0.2 for x in xs], [helped_hurt[k][0] for k in ks], 0.4, label="helped", color="#3E9A6E")
    ax.bar([x + 0.2 for x in xs], [helped_hurt[k][1] for k in ks], 0.4, label="hurt", color="#B5533C")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("# queries")
    ax.set_title("Reranker vs hybrid: queries helped / hurt")
    ax.legend()

    # panel 2: alpha sweep
    ax = axes[1]
    alphas = [row["alpha"] for row in sweep_rows]
    for metric, label in (("hit@1", "Hit@1"), ("hit@5", "Hit@5"), ("hit@10", "Hit@10")):
        ax.plot(alphas, [100 * row[metric] for row in sweep_rows], marker="o", label=label)
    ax.axvline(fusion_alpha, color="gray", linestyle=":", label=f"alpha used ({fusion_alpha})")
    ax.set_xlabel("alpha (0 = hybrid only, 1 = reranker only)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("Blending reranker + hybrid scores")
    ax.legend()

    # panel 3: rank histogram of first relevant doc
    ax = axes[2]
    buckets = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5),
               ("6-10", 6, 10), ("11-20", 11, 20), ("21+", 21, 10**9)]
    labels = [b[0] for b in buckets] + ["not in pool"]

    def count(order_key):
        counts = [0] * len(labels)
        for r in records:
            rank = _first_rank(r[order_key], r["relevant"])
            if rank is None:
                counts[-1] += 1
                continue
            for i, (_, lo, hi) in enumerate(buckets):
                if lo <= rank <= hi:
                    counts[i] += 1
                    break
        return counts

    h_counts, r_counts = count("pool_ids"), count("rerank_order")
    xs = range(len(labels))
    ax.bar([x - 0.2 for x in xs], h_counts, 0.4, label="hybrid", color="#3E9A6E")
    ax.bar([x + 0.2 for x in xs], r_counts, 0.4, label="reranked", color="#D98E3B")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_xlabel("rank of first relevant doc")
    ax.set_ylabel("# queries")
    ax.set_title("Where the right answer lands")
    ax.legend()

    plt.tight_layout()
    path = out_dir / "diagnostics.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved diagnostics chart -> {path}")