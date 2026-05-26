"""
utils.py — 汎用ユーティリティ
"""
import re
import pandas as pd
from config import DECAY


def wt(i: int) -> float:
    return DECAY[i] if i < len(DECAY) else 0.05


def fc(cols, *kws):
    """列名リストからキーワードに部分一致する最初の列を返す"""
    for kw in kws:
        for c in cols:
            if kw in c:
                return c
    return None


def find_col(df: pd.DataFrame, keywords: list) -> str | None:
    for kw in keywords:
        for c in df.columns:
            if kw in c:
                return c
    return None


def make_unique_columns(cols: list) -> list:
    seen, out = {}, []
    for c in cols:
        if c not in seen:
            seen[c] = 0; out.append(c)
        else:
            seen[c] += 1; out.append(f"{c}_{seen[c]}")
    return out


def parse_time(val) -> float | None:
    s = str(val).strip()
    m = re.match(r"(\d+):(\d+\.\d+)", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.match(r"(\d{2,3}\.\d)", s)
    return float(m.group(1)) if m else None


def clean_order(val) -> int | None:
    """着順列から数値を取り出す（取消・中止など除去）"""
    raw = re.sub(r"[^\d]", "", str(val).strip())
    n = pd.to_numeric(raw, errors="coerce")
    if pd.isna(n) or n <= 0:
        return None
    return int(n)


def normalize_scores(scores: list) -> list:
    """レース内で 0〜100 に正規化"""
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [50.0] * len(scores)
    return [(s - mn) / (mx - mn) * 100.0 for s in scores]
