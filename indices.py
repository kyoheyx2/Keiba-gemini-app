"""
indices.py — 7指数の計算
  ① スピード指数
  ② 能力指数（スピード指数ベース × クラス × 着順 × 枠番）
  ③ EB指数（Beta-Binomial shrinkage）
  ④ Base指数（コース・距離適性）
  ⑤ 騎手指数（netkeiba リアルタイム取得）
  ⑥ 前走間隔スコア
  ⑦ 上がり3F指数
"""
import io, re, time
import pandas as pd
import streamlit as st

from config import (
    BASE_TIMES, DIST_IDX, TRACK_COND_ADJ,
    INNER_FAVOR_COURSES, FLAT_COURSES,
    LEVEL_MULT, EB_ALPHA, EB_DEFAULT,
)
from utils import fc, wt, parse_time, clean_order, make_unique_columns, find_col
from scraper import get_soup, wait_for_page


# ═══════════════════════════════════════════════════════════
#  共通ヘルパー
# ═══════════════════════════════════════════════════════════
def get_base_time(track: str, surface: str, dist: int) -> float:
    key = (track, surface, dist)
    if key in BASE_TIMES:
        return BASE_TIMES[key]
    pool = sorted(
        [(k[2], v) for k, v in BASE_TIMES.items()
         if k[0] in (track, "汎用") and k[1] == surface]
    )
    if not pool:
        pool = sorted([(k[2], v) for k, v in BASE_TIMES.items() if k[1] == surface])
    if not pool:
        return 95.0
    dists, times = [p[0] for p in pool], [p[1] for p in pool]
    if dist <= dists[0]:  return times[0]
    if dist >= dists[-1]: return times[-1]
    for i in range(len(dists) - 1):
        if dists[i] <= dist <= dists[i + 1]:
            r = (dist - dists[i]) / (dists[i + 1] - dists[i])
            return times[i] + r * (times[i + 1] - times[i])
    return times[-1]


def get_dist_index(dist: int) -> float:
    keys = sorted(DIST_IDX)
    if dist in DIST_IDX:
        return DIST_IDX[dist]
    lo = max((k for k in keys if k <= dist), default=keys[0])
    hi = min((k for k in keys if k >= dist), default=keys[-1])
    if lo == hi:
        return DIST_IDX[lo]
    r = (dist - lo) / (hi - lo)
    return DIST_IDX[lo] + r * (DIST_IDX[hi] - DIST_IDX[lo])


def get_frame_adj(gate_num: int, track: str, surface: str, dist: int) -> float:
    if gate_num <= 0:
        return 1.0
    key = (track, surface, dist)
    if key in INNER_FAVOR_COURSES:
        if gate_num <= 3:  return 1.08
        if gate_num <= 6:  return 1.04
        if gate_num <= 12: return 1.00
        if gate_num <= 15: return 0.94
        return 0.90
    elif key in FLAT_COURSES:
        if gate_num <= 6:  return 1.02
        if gate_num <= 12: return 1.00
        return 0.97
    else:
        if gate_num <= 3:  return 1.05
        if gate_num <= 6:  return 1.02
        if gate_num <= 12: return 1.00
        if gate_num <= 15: return 0.96
        return 0.92


# ═══════════════════════════════════════════════════════════
#  ① スピード指数（西田式）
# ═══════════════════════════════════════════════════════════
def calc_speed_index(past_df: pd.DataFrame,
                     race_track="汎用", race_surface="芝",
                     race_dist=0, burden=55.0) -> float:
    df   = past_df.copy().reset_index(drop=True)
    cols = df.columns.tolist()
    time_col  = fc(cols, "タイム")
    dist_col  = fc(cols, "距離")
    cond_col  = fc(cols, "馬場", "状態")
    burd_col  = fc(cols, "斤量")
    track_col = fc(cols, "競馬場", "開催場", "場名")
    scores = []
    for i, (_, row) in enumerate(df.iterrows()):
        t = parse_time(row.get(time_col)) if time_col else None
        if not t:
            continue
        d_raw = (str(row.get(dist_col, ""))
                 .replace("m","").replace("芝","").replace("ダ","").strip())
        try:    d = int(float(d_raw))
        except: d = race_dist if race_dist > 0 else 1600
        surf  = ("芝" if "芝" in str(row.get(dist_col,""))
                 else "ダ" if "ダ" in str(row.get(dist_col,""))
                 else race_surface)
        trk   = str(row.get(track_col,"")).strip() if track_col else race_track
        base      = get_base_time(trk, surf, d)
        dist_idx  = get_dist_index(d)
        cond_val  = str(row.get(cond_col,"良")).strip() if cond_col else "良"
        track_adj = TRACK_COND_ADJ.get(cond_val, 0.0)
        burd_val  = float(pd.to_numeric(row.get(burd_col, 55), errors="coerce") or 55)
        burd_adj  = (burd_val - 55) * 2.0
        spd = (base - t) * dist_idx + track_adj + burd_adj + 80
        scores.append(spd * wt(i))
    tw = sum(wt(i) for i in range(len(scores)))
    return sum(scores) / tw if tw > 0 else 80.0


# ═══════════════════════════════════════════════════════════
#  ② 能力指数
# ═══════════════════════════════════════════════════════════
def get_level_mult(race_name: str) -> float:
    if not isinstance(race_name, str) or not race_name:
        return 1.0
    for k, v in LEVEL_MULT.items():
        if k in race_name:
            return v
    return 1.0

def get_place_mult(order: int) -> float:
    if order == 1: return 1.00
    if order == 2: return 0.90
    if order == 3: return 0.80
    if order <= 5: return 0.65
    return 0.50

def calc_ability_index(past_df: pd.DataFrame,
                       gate_num=0, race_track="汎用",
                       race_surface="芝", race_dist=1600) -> float:
    df   = past_df.copy().reset_index(drop=True)
    cols = df.columns.tolist()
    time_col  = fc(cols, "タイム")
    dist_col  = fc(cols, "距離")
    cond_col  = fc(cols, "馬場", "状態")
    burd_col  = fc(cols, "斤量")
    track_col = fc(cols, "競馬場", "開催場", "場名")
    ord_col   = fc(cols, "着順")
    race_col  = fc(cols, "レース名", "競走名", "レース", "競走", "条件", "クラス")
    frame_adj = get_frame_adj(gate_num, race_track, race_surface, race_dist)
    scores = []
    for i, (_, row) in enumerate(df.iterrows()):
        t = parse_time(row.get(time_col)) if time_col else None
        if not t:
            continue
        d_raw = (str(row.get(dist_col,"")).replace("m","")
                 .replace("芝","").replace("ダ","").strip()) if dist_col else ""
        try:    d = int(float(d_raw))
        except: d = race_dist if race_dist > 0 else 1600
        surf = ("芝" if "芝" in str(row.get(dist_col,""))
                else "ダ" if "ダ" in str(row.get(dist_col,""))
                else race_surface)
        trk      = str(row.get(track_col,"")).strip() if track_col else race_track
        cond_val = str(row.get(cond_col,"良")).strip() if cond_col else "良"
        burd_val = float(pd.to_numeric(row.get(burd_col,55), errors="coerce") or 55)
        base      = get_base_time(trk, surf, d)
        dist_idx  = get_dist_index(d)
        track_adj = TRACK_COND_ADJ.get(cond_val, 0.0)
        burd_adj  = (burd_val - 55) * 2.0
        speed_idx = (base - t) * dist_idx + track_adj + burd_adj + 80
        race_name = str(row.get(race_col,"")).strip() if race_col else ""
        level_m   = get_level_mult(race_name)
        order_int = clean_order(row.get(ord_col,"")) or 6
        place_m   = get_place_mult(max(1, order_int))
        scores.append(speed_idx * level_m * place_m * frame_adj * wt(i))
    if not scores:
        return 50.0
    tw = sum(wt(i) for i in range(len(scores)))
    return sum(scores) / tw if tw > 0 else 50.0


# ═══════════════════════════════════════════════════════════
#  ③ EB指数（Beta-Binomial shrinkage）
# ═══════════════════════════════════════════════════════════
def calc_eb_index(past_df: pd.DataFrame, mu_prior=EB_DEFAULT) -> float:
    df   = past_df.copy().reset_index(drop=True)
    cols = df.columns.tolist()
    ord_col = None
    for kw in ["着順", "着"]:
        for c in cols:
            if kw in c:
                ord_col = c
                break
        if ord_col:
            break
    if not ord_col:
        return mu_prior * 100
    cleaned = (df[ord_col].astype(str).str.strip()
               .str.replace(r"[^\d]", "", regex=True))
    orders = pd.to_numeric(cleaned, errors="coerce").dropna()
    orders = orders[orders > 0]
    if orders.empty:
        return mu_prior * 100
    success = sum(
        1.0 if o == 1 else 0.6 if o == 2 else 0.3 if o == 3 else 0.0
        for o in orders
    )
    return ((success + EB_ALPHA * mu_prior) / (len(orders) + EB_ALPHA)) * 100


def update_mu_prior(all_past: list) -> float:
    ts, tt = 0.0, 0
    for pdf in all_past:
        ord_col = None
        for kw in ["着順", "着"]:
            for c in pdf.columns:
                if kw in c:
                    ord_col = c
                    break
            if ord_col:
                break
        if not ord_col:
            continue
        cleaned = (pdf[ord_col].astype(str).str.strip()
                   .str.replace(r"[^\d]", "", regex=True))
        for o in pd.to_numeric(cleaned, errors="coerce").dropna():
            if o > 0:
                tt += 1
                if o <= 3:
                    ts += 1.0
    if tt < 10:
        return EB_DEFAULT
    return float(min(max(ts / tt, 0.05), 0.50))


# ═══════════════════════════════════════════════════════════
#  ④ Base指数（コース・距離適性）
# ═══════════════════════════════════════════════════════════
def calc_base_index(past_df: pd.DataFrame,
                    race_track="汎用", race_surface="芝",
                    race_dist=0, burden=55.0) -> float:
    """
    コース・距離適性スコアを 0〜100 スケールで返す。
    （旧実装は 0〜10 スケールだったため他指数と不均一だった）
    """
    df   = past_df.copy()
    cols = df.columns.tolist()
    dist_col  = fc(cols, "距離")
    ord_col   = fc(cols, "着順")
    track_col = fc(cols, "競馬場", "開催場", "場名")

    # 着順列をクリーニング
    if ord_col:
        df[ord_col] = pd.to_numeric(
            df[ord_col].astype(str).str.replace(r"[^\d]", "", regex=True),
            errors="coerce"
        )

    # ── コース適性スコア（0〜100）──
    course_score = 50.0   # デフォルト中央値
    if dist_col and ord_col:
        df["_surf"] = df[dist_col].astype(str).str.contains(race_surface, na=False)
        if track_col:
            df["_track"] = df[track_col].astype(str).str.contains(race_track, na=False)
            matched = df[df["_surf"] & df["_track"]]
        else:
            matched = df[df["_surf"]]
        if not matched.empty:
            orders = pd.to_numeric(matched[ord_col], errors="coerce").dropna()
            orders = orders[orders > 0]
            if not orders.empty:
                # 平均着順 1→100点, 5→60点, 10→10点 に線形変換
                avg = orders.mean()
                course_score = float(max(min(110.0 - avg * 10.0, 100.0), 10.0))

    # ── 距離適性スコア（0〜100）──
    dist_score = 50.0   # デフォルト中央値
    if dist_col and race_dist > 0:
        df["_d"]    = pd.to_numeric(
            df[dist_col].astype(str).str.extract(r"(\d{3,4})")[0], errors="coerce"
        )
        df["_diff"] = (df["_d"] - race_dist).abs()
        df["_ord"]  = (pd.to_numeric(df[ord_col], errors="coerce").fillna(9.0)
                       if ord_col else 9.0)
        df["_ord"]  = df["_ord"].where(df["_ord"] > 0, 9.0)
        near = df[df["_diff"] <= 300]
        if not near.empty:
            bonus = 15.0 if ((near["_diff"] <= 200) & (near["_ord"] <= 3)).any() else 0.0
            avg = near["_ord"].mean()
            dist_score = float(max(min(110.0 - avg * 10.0 + bonus, 100.0), 10.0))

    # 斤量ペナルティ（55kgを基準に重いほど減点）
    burd_pen = max((burden - 55.0) * 3.0, 0.0)   # 1kg重いごとに-3点
    raw = course_score * 0.5 + dist_score * 0.5 - burd_pen
    return float(max(min(raw, 100.0), 0.0))


# ═══════════════════════════════════════════════════════════
#  ⑤ 騎手指数（直近レース結果からEB方式で計算）
#
#  アプローチ：
#  騎手IDのレース結果一覧ページ(/jockey/{id}/)から
#  直近50走の着順を取得し、EB指数(Beta-Binomial)と同じ方法で
#  勝率・連対率・複勝率を計算して指数化する。
#  これにより「勝率」列の解析問題を完全に回避する。
# ═══════════════════════════════════════════════════════════

def _parse_jockey_race_results(driver, jockey_id: str) -> list[int]:
    """
    /jockey/{id}/ の直近レース結果から着順リストを取得する。
    着順列に含まれる数値のみを抽出して返す。
    """
    urls = [
        f"https://db.netkeiba.com/jockey/result/recent/{jockey_id}/",
        f"https://db.netkeiba.com/jockey/{jockey_id}/",
    ]
    for url in urls:
        try:
            driver.get(url)
            wait_for_page(driver, "table", timeout=8, fallback=2)
            soup = get_soup(driver)
            tables = soup.find_all("table")
            if not tables:
                continue

            for t in sorted(tables,
                            key=lambda x: len(x.find_all("tr")), reverse=True):
                try:
                    df = pd.read_html(io.StringIO(str(t)))[0]
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [
                            "_".join(str(c) for c in col if str(c) != "nan").strip()
                            for col in df.columns
                        ]
                    else:
                        df.columns = [str(c).strip() for c in df.columns]
                    df.columns = make_unique_columns(df.columns.tolist())

                    # 「着順」または「順位」列を探す
                    ord_col = fc(df.columns.tolist(), "着順", "順位", "着")
                    if not ord_col:
                        continue

                    # 数値のみ抽出（取消・中止・除外などを除去）
                    cleaned = (
                        df[ord_col].astype(str)
                        .str.replace(r"[^\d]", "", regex=True)
                        .str.strip()
                    )
                    orders = pd.to_numeric(cleaned, errors="coerce").dropna()
                    orders = orders[(orders >= 1) & (orders <= 28)].astype(int).tolist()

                    if len(orders) >= 5:
                        return orders[:50]  # 直近50走
                except Exception:
                    continue
        except Exception:
            continue
    return []


def calc_jockey_score(driver, jockey_id: str,
                      race_track: str, race_surface: str) -> float:
    """
    騎手の直近レース着順からEB方式で指数を計算する。

    取得方法:
      /jockey/result/recent/{id}/ の着順列を直接読む
      → Beta-Binomial shrinkage で勝率・連対率・複勝率を安定推定
      → 指数化（基準：勝率10%・連対率20%・複勝率33% → 50点）

    成績テーブルの「勝率」列解析を完全に回避するため安定して動作する。
    """
    if not jockey_id:
        return 50.0

    cache     = st.session_state.horse_cache
    cache_key = f"jockey_{jockey_id}_{race_track}_{race_surface}"
    if cache_key in cache:
        return cache[cache_key]

    # 着順リスト取得
    orders = _parse_jockey_race_results(driver, jockey_id)

    if not orders:
        cache[cache_key] = 50.0
        return 50.0

    n = len(orders)

    # ── Beta-Binomial shrinkage ──────────────────────────────
    # 過去成績が少ない騎手ほど平均値に引き寄せる
    ALPHA = 15.0
    WIN_PRIOR    = 0.10   # JRA平均勝率
    RENTAI_PRIOR = 0.20   # JRA平均連対率
    FUKUSHO_PRIOR= 0.33   # JRA平均複勝率

    wins     = sum(1   for o in orders if o == 1)
    rentai   = sum(1   for o in orders if o <= 2)
    fukusho  = sum(1   for o in orders if o <= 3)

    win_r     = (wins    + ALPHA * WIN_PRIOR)     / (n + ALPHA) * 100
    rentai_r  = (rentai  + ALPHA * RENTAI_PRIOR)  / (n + ALPHA) * 100
    fukusho_r = (fukusho + ALPHA * FUKUSHO_PRIOR) / (n + ALPHA) * 100

    # 指数化：基準値(10*2.5 + 20*1.0 + 33*0.5 = 61.5) → 50点
    raw   = win_r * 2.5 + rentai_r * 1.0 + fukusho_r * 0.5
    score = float(min(max(raw / 61.5 * 50.0, 0.0), 100.0))

    # デバッグ保存
    cache[f"jockey_stat_{jockey_id}"] = {
        "races":      n,
        "wins":       wins,
        "rentai":     rentai,
        "fukusho":    fukusho,
        "win_r":      round(win_r, 1),
        "rentai_r":   round(rentai_r, 1),
        "fukusho_r":  round(fukusho_r, 1),
        "score":      round(score, 1),
    }
    cache[cache_key] = score
    return score

# ═══════════════════════════════════════════════════════════
#  ⑥ 前走間隔スコア
# ═══════════════════════════════════════════════════════════
def calc_interval_score(past_df: pd.DataFrame) -> float:
    df   = past_df.copy().reset_index(drop=True)
    cols = df.columns.tolist()
    date_col = fc(cols, "日付", "年月日", "date")
    if not date_col or len(df) < 1:
        return 50.0
    try:
        date_str = str(df.iloc[0][date_col]).strip()
        date_str = re.sub(r"[年月]", "/", date_str).replace("日", "")
        last_date = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(last_date):
            return 50.0
        weeks = (pd.Timestamp.today() - last_date).days / 7.0
        if weeks < 0:   return 50.0
        if weeks <= 2:  return 38.0   # 中1〜2週：短期疲労リスク
        if weeks <= 4:  return 55.0   # 中3〜4週：標準
        if weeks <= 8:  return 62.0   # 中5〜8週：叩き2走目・好仕上がり
        if weeks <= 15: return 50.0   # 中9〜15週：やや間隔空く
        return 35.0                   # 中16週以上：長期休養明け
    except Exception:
        return 50.0


# ═══════════════════════════════════════════════════════════
#  ⑦ 上がり3F指数
# ═══════════════════════════════════════════════════════════
def calc_finish3f_score(past_df: pd.DataFrame) -> float:
    df   = past_df.copy().reset_index(drop=True)
    cols = df.columns.tolist()
    finish_col = fc(cols, "上り", "上がり", "finish", "F3")
    if not finish_col:
        return 50.0
    scores = []
    for i, (_, row) in enumerate(df.iterrows()):
        val = str(row.get(finish_col, "")).strip()
        try:
            f = float(val)
            if f <= 0 or f > 45:
                continue
            # 33.5秒基準：1秒遅いごとに-10点
            score = 80.0 - (f - 33.5) * 10.0
            score = max(10.0, min(100.0, score))
            scores.append(score * wt(i))
        except ValueError:
            continue
    if not scores:
        return 50.0
    tw = sum(wt(i) for i in range(len(scores)))
    return sum(scores) / tw if tw > 0 else 50.0