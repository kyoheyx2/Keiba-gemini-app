"""
optimizer.py — ウェイト自動最適化エンジン
過去走データから「指数順位 vs 実際の着順」のスピアマン順位相関を
最大化するウェイトをNelder-Mead（ランダム再起動付き）で求める。
"""
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.optimize import minimize
from scipy.stats import spearmanr

from config import DEFAULT_WEIGHTS, MIN_SAMPLES_FOR_OPT
from utils import normalize_scores


def calc_total_scores(samples: list[dict], weights: list[float]) -> list[float]:
    """レース内正規化 + ウェイト適用で総合スコアを計算"""
    race_groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        race_groups[s["race_key"]].append(i)

    totals = [0.0] * len(samples)
    idx_keys = ["s1","s2","s3","s4","s5","s6","s7"]
    for idxs in race_groups.values():
        for col, w in zip(idx_keys, weights):
            vals   = [samples[i].get(col, 50.0) for i in idxs]
            normed = normalize_scores(vals)
            for j, idx in enumerate(idxs):
                totals[idx] += normed[j] * w
    return totals


def optimize_weights(samples: list[dict]) -> list[float]:
    """
    スピアマン順位相関を最大化するウェイトを求める。
    - log空間（softmax）で最適化することで合計=1を自然に満たす
    - Nelder-Mead + ランダム初期値50回でグローバル最適を探索
    """
    if len(samples) < MIN_SAMPLES_FOR_OPT:
        return DEFAULT_WEIGHTS

    actual_ranks = [s["actual_rank"] for s in samples]
    race_groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        race_groups[s["race_key"]].append(i)

    def objective(log_w):
        w = np.exp(log_w) / np.sum(np.exp(log_w))
        totals = calc_total_scores(samples, w.tolist())
        corr_sum = 0.0
        for idxs in race_groups.values():
            if len(idxs) < 3:
                continue
            t = [totals[i] for i in idxs]
            r = [actual_ranks[i] for i in idxs]
            c, _ = spearmanr(t, r)
            if not np.isnan(c):
                corr_sum += c
        return corr_sum  # 最小化 → 負方向に大きく = 着順相関が強い

    best_val = float("inf")
    best_w   = DEFAULT_WEIGHTS[:]
    rng = np.random.default_rng(42)

    for _ in range(50):
        x0 = rng.dirichlet([1] * len(DEFAULT_WEIGHTS))
        try:
            res = minimize(
                objective, np.log(x0),
                method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-8},
            )
            if res.fun < best_val:
                best_val = res.fun
                w = np.exp(res.x) / np.sum(np.exp(res.x))
                best_w = w.tolist()
        except Exception:
            continue

    return [round(w, 4) for w in best_w]


def accumulate_samples(results_raw: list[dict], race_key: str,
                       past_map: dict) -> list[dict]:
    """
    今回計算したレースの各馬について
    {s1〜s7, actual_rank, race_key} をサンプルに蓄積する。
    actual_rank は過去走テーブルの直前レース着順を正解ラベルとして使用。
    """
    new_samples = []
    for r in results_raw:
        hname = r["馬名"]
        past  = past_map.get(hname)
        if past is None or past.empty:
            continue
        ord_col = None
        for kw in ["着順", "着"]:
            for c in past.columns:
                if kw in c:
                    ord_col = c
                    break
            if ord_col:
                break
        if not ord_col:
            continue
        order_raw = re.sub(r"[^\d]", "", str(past.iloc[0][ord_col]).strip())
        actual = pd.to_numeric(order_raw, errors="coerce")
        if pd.isna(actual) or actual <= 0:
            continue
        new_samples.append({
            "race_key":    race_key,
            "horse":       hname,
            "s1": r["_s1"], "s2": r["_s2"], "s3": r["_s3"], "s4": r["_s4"],
            "s5": r["_s5"], "s6": r["_s6"], "s7": r["_s7"],
            "actual_rank": int(actual),
        })
    return new_samples
