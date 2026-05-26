"""
app.py — Streamlit UI メイン
競馬予想システム: 7指数統合モデル × 競馬場別ウェイト × AI分析
"""
import re
import io
import pandas as pd
import streamlit as st
from collections import defaultdict
from datetime import datetime

from config import (
    PLACE_MAP, DEFAULT_WEIGHTS, MIN_RACES_FOR_OPT,
    MIN_SAMPLES_FOR_OPT, get_course_weights,
)
from utils import normalize_scores, make_unique_columns, find_col
from scraper import (
    get_driver, fetch_race_list, get_horse_ids_from_page,
    get_past_races, get_training_text, parse_shutuba_table,
    parse_race_info, get_soup, wait_for_page,
)
from indices import (
    calc_speed_index, calc_ability_index, calc_eb_index,
    calc_base_index, calc_jockey_score,
    calc_interval_score, calc_finish3f_score, update_mu_prior,
)
from optimizer import (
    calc_total_scores, optimize_weights, accumulate_samples,
)
from ai import (
    call_gemini, build_analysis_prompt, parse_ai_response,
    render_ai_result, get_gemini_key,
)

# ═══════════════════════════════════════════════════════════
#  ページ設定
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🏇 競馬予想アプリ",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
.block-container{padding:2.5rem 1rem 2rem!important;max-width:100%!important}
h1{font-size:1.5rem!important}
div.stButton>button{width:100%!important;min-height:3rem!important;font-size:1rem!important;border-radius:8px!important}
div.stButton>button[kind="primary"]{min-height:3.4rem!important;font-size:1.05rem!important}
div[data-testid="stDataFrame"]{overflow-x:auto!important}
div[data-testid="stDataFrame"] table{font-size:.85rem!important}
</style>
""", unsafe_allow_html=True)

st.title("🏇 競馬予想アプリ — 7指数×競馬場別ウェイト×AI分析")
st.caption("スピード・能力・EB・Base・騎手・前走間隔・上がり3F（競馬場×馬場状態で自動切替）")

# ═══════════════════════════════════════════════════════════
#  セッション初期化
# ═══════════════════════════════════════════════════════════
for key, default in [
    ("race_links",     []),
    ("horse_cache",    {}),
    ("opt_weights",    None),
    ("opt_samples",    []),
    ("opt_race_count", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ═══════════════════════════════════════════════════════════
#  サイドバー
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 設定")

    # Gemini APIキー
    api_key_input = st.text_input(
        "Gemini APIキー",
        value=get_gemini_key(),
        type="password",
        help=".streamlit/secrets.toml に GEMINI_API_KEY を設定するか、ここに入力",
    )
    GEMINI_API_KEY = api_key_input.strip() or get_gemini_key()

    st.markdown("---")

    # 馬場状態（ウェイト切替に使用）
    st.markdown("**🌧️ 当日馬場状態**")
    baba_select = st.selectbox(
        "馬場状態",
        ["良","稍重","重","不良"],
        help="競馬場×馬場状態でウェイトが自動切替されます"
    )

    st.markdown("---")

    # 最適化ウェイト表示
    st.markdown("**🔬 実績ベース自動最適化ウェイト**")
    opt_w     = st.session_state.opt_weights
    n_samples = len(st.session_state.opt_samples)
    n_races   = st.session_state.opt_race_count
    use_opt   = False

    if opt_w:
        st.caption(f"蓄積: {n_races}レース / {n_samples}サンプル")
        labels = "①②③④⑤⑥⑦"
        w_str  = " ".join(f"{labels[j]}{opt_w[j]:.2f}" for j in range(len(opt_w)))
        st.caption(w_str)
        use_opt = st.checkbox("✅ 最適化ウェイトを優先する", value=False,
                              help="ONにすると競馬場別ウェイトより最適化ウェイトが優先されます")
    else:
        st.caption(f"蓄積中... ({n_samples}サンプル)")
        st.caption("10レース以上計算すると自動最適化されます")

    if st.button("🗑️ 蓄積データをリセット", use_container_width=True):
        st.session_state.opt_samples    = []
        st.session_state.opt_weights    = None
        st.session_state.opt_race_count = 0
        st.rerun()

    st.markdown("---")
    if st.checkbox("🔍 デバッグ情報"):
        # 過去走テーブル列名
        debug_cols = st.session_state.horse_cache.get("debug_cols")
        if debug_cols:
            st.markdown("**過去走テーブルの列名:**")
            st.code("\n".join(debug_cols))
        # 騎手テーブル列名（取得失敗した場合のデバッグ用）
        jockey_debug = {k: v for k, v in st.session_state.horse_cache.items()
                        if k.startswith("jockey_debug_")}
        if jockey_debug:
            st.markdown("**騎手テーブルの列名:**")
            for k, v in list(jockey_debug.items())[:3]:
                st.caption(k.replace("jockey_debug_","ID:"))
                st.code(str(v[:15]))
        # 騎手EB計算結果
        jockey_stats = {k: v for k, v in st.session_state.horse_cache.items()
                        if k.startswith("jockey_stat_")}
        if jockey_stats:
            st.markdown("**騎手指数（EB方式）:**")
            for k, v in list(jockey_stats.items())[:8]:
                st.caption(
                    f"ID:{k.replace('jockey_stat_','')} "
                    f"{v.get('races',0)}走 "
                    f"{v.get('wins',0)}勝 "
                    f"勝率{v.get('win_r',0)}% "
                    f"連{v.get('rentai_r',0)}% "
                    f"複{v.get('fukusho_r',0)}% "
                    f"→ {v.get('score',50)}"
                )

# ═══════════════════════════════════════════════════════════
#  ウェイト取得（競馬場別 or 最適化）
# ═══════════════════════════════════════════════════════════
def resolve_weights(race_track: str) -> tuple[dict, str]:
    """
    使用ウェイトと説明文を返す。
    最適化ウェイト優先フラグがONで最適化済みなら最適化ウェイトを返す。
    そうでなければ競馬場×馬場状態別ウェイトを返す。
    """
    if use_opt and opt_w:
        w = {
            "speed":    opt_w[0], "ability": opt_w[1],
            "eb":       opt_w[2], "base":    opt_w[3],
            "jockey":   opt_w[4], "interval":opt_w[5],
            "finish3f": opt_w[6],
        }
        labels = "①②③④⑤⑥⑦"
        desc = "最適化ウェイト " + " ".join(
            f"{labels[j]}{opt_w[j]:.2f}" for j in range(len(opt_w))
        )
    else:
        w    = get_course_weights(race_track, baba_select)
        desc = (f"競馬場別ウェイト ({race_track}/{baba_select}) "
                f"速{w['speed']:.2f} 能{w['ability']:.2f} "
                f"EB{w['eb']:.2f} 騎{w['jockey']:.2f} 上{w['finish3f']:.2f}")
    return w, desc


def calc_total(scores: dict, w: dict, growth: float) -> float:
    """正規化済みスコアとウェイトから総合指数を計算"""
    return (
        scores["s1"] * w["speed"]    +
        scores["s2"] * w["ability"]  +
        scores["s3"] * w["eb"]       +
        scores["s4"] * w["base"]     +
        scores["s5"] * w["jockey"]   +
        scores["s6"] * w["interval"] +
        scores["s7"] * w["finish3f"] +
        growth
    )


def ranked_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out.index = range(1, len(out)+1)
    out.index.name = "順位"
    return out


def recommend_tickets(result_df: pd.DataFrame):
    if len(result_df) < 2:
        return
    names = result_df["馬名"].head(3).tolist()
    gap   = float(result_df.iloc[0]["総合指数"]) - float(result_df.iloc[1]["総合指数"])

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("◎ 本命", names[0])
    with c2: st.metric("○ 対抗", names[1])
    with c3: st.metric("△ 穴馬(3位)", names[2] if len(names)>2 else "−")

    tickets = [
        {"券種":"単勝",   "組み合わせ": names[0],                      "比率":"20%", "理由":"指数1位"},
        {"券種":"馬連",   "組み合わせ": f"{names[0]}−{names[1]}",      "比率":"40%", "理由":"指数上位2頭"},
        {"券種":"ワイド", "組み合わせ": f"{names[0]}−{names[1]}",      "比率":"20%", "理由":"堅め"},
    ]
    if len(names) >= 3:
        tickets += [
            {"券種":"ワイド", "組み合わせ": f"{names[0]}−{names[2]}", "比率":"10%", "理由":"穴絡み"},
            {"券種":"三連複", "組み合わせ": f"{names[0]}−{names[1]}−{names[2]}", "比率":"10%", "理由":"TOP3"},
        ]
    icons = {"単勝":"🥇","馬連":"🔗","ワイド":"🎯","三連複":"🎰"}
    for t in tickets:
        note = f"　※差{gap:.1f}pt" if t["券種"]=="単勝" and gap>10 else ""
        st.success(
            f"{icons.get(t['券種'],'🎫')} **{t['券種']}**　"
            f"{t['組み合わせ']}　{t['比率']}　{t['理由']}{note}"
        )


# ═══════════════════════════════════════════════════════════
#  メイン UI
# ═══════════════════════════════════════════════════════════
selected_date = st.date_input("📅 分析日付", value=datetime.today().date())
kaisai_date   = selected_date.strftime("%Y%m%d")

# ── ステップ1: レース一覧取得 ─────────────────────────────────
if st.button("🔍 レース一覧を取得", type="primary"):
    with st.spinner("netkeiba からレース一覧を取得中..."):
        try:
            driver = get_driver()
            st.session_state.horse_cache = {}
            links  = fetch_race_list(driver, kaisai_date)
            st.session_state.race_links = links
            st.success(f"✅ {len(links)} レースを取得しました！")
        except Exception as e:
            st.error(f"取得エラー: {e}")

# ── ステップ2: レース選択 ─────────────────────────────────────
if st.session_state.race_links:
    st.subheader("🏇 分析するレースを選択")

    grouped = defaultdict(list)
    for name, url in st.session_state.race_links:
        m = re.search(r"race_id=(\d{4})(\d{2})\d+", url)
        venue = PLACE_MAP.get(m.group(2) if m else "", "その他")
        grouped[venue].append((name, url))

    venues_order = [
        "東京","京都","阪神","中山","新潟","小倉",
        "福島","中京","札幌","函館","その他"
    ]
    venues = [v for v in venues_order if v in grouped]
    tabs   = st.tabs(venues)

    selected_races = []
    for tab, venue in zip(tabs, venues):
        with tab:
            st.markdown(f"**🏟️ {venue}**")
            for name, url in grouped[venue]:
                m_id = re.search(r"race_id=(\d+)", url)
                cb_key = f"cb_{m_id.group(1) if m_id else name}"
                if st.checkbox(name, key=cb_key):
                    selected_races.append((name, url))

    if selected_races:
        st.info(f"✅ {len(selected_races)} レース選択中")

    # ── ステップ3: 計算・分析 ─────────────────────────────────
    if st.button("📊 選択レースの指数を計算・AI分析", type="primary"):
        if not selected_races:
            st.warning("レースを選択してください")
        else:
            driver = get_driver()

            for race_name, url in selected_races:
                with st.spinner(f"⏳ {race_name} 処理中..."):
                    try:
                        driver.get(url)
                        wait_for_page(driver, "table", timeout=10, fallback=3)
                        soup = get_soup(driver)

                        # 出馬表テーブル取得
                        table = None
                        for kw in ["shutuba_table","Shutuba_Table","race_table_01"]:
                            table = soup.find(
                                "table",
                                class_=lambda x: x and kw in " ".join(x) if x else False
                            )
                            if table: break
                        if table is None:
                            all_t = soup.find_all("table")
                            if all_t:
                                table = max(all_t, key=lambda t: len(t.find_all("tr")))
                        if table is None:
                            st.error(f"{race_name}：テーブルが見つかりません")
                            continue

                        # 馬ID・レース情報・出馬表パース
                        horse_info = get_horse_ids_from_page(driver, soup)
                        if not horse_info:
                            st.error(f"{race_name}：馬IDが取得できません")
                            continue

                        race_track, race_surface, race_distance = parse_race_info(soup, url)
                        shutuba     = parse_shutuba_table(table, soup)
                        w, w_desc   = resolve_weights(race_track)
                        mu_prior    = st.session_state.horse_cache.get("mu_prior", 0.18)

                        st.caption(f"  → {len(horse_info)}頭 / {race_track} {race_surface}{race_distance}m")

                        # ── 各馬の指数計算 ──
                        results  = []
                        past_map = {}
                        progress = st.progress(0)
                        status   = st.empty()

                        for idx, (hname, horse_id) in enumerate(horse_info):
                            progress.progress((idx+1) / len(horse_info))
                            status.caption(f"処理中: {hname} ({idx+1}/{len(horse_info)})")

                            past = get_past_races(driver, horse_id)
                            if past is None or past.empty:
                                continue

                            info       = shutuba.get(hname, {})
                            burden_val = info.get("burden", 55.0)
                            gate_num   = info.get("gate_num", 0)
                            jockey_id  = info.get("jockey_id", "")
                            jockey_name= info.get("jockey_name", "未定")
                            age_str    = info.get("age", "")

                            s1 = calc_speed_index(past, race_track, race_surface, race_distance, burden_val)
                            s2 = calc_ability_index(past, gate_num, race_track, race_surface, race_distance)
                            s3 = calc_eb_index(past, mu_prior)
                            s4 = calc_base_index(past, race_track, race_surface, race_distance, burden_val)
                            s5 = calc_jockey_score(driver, jockey_id, race_track, race_surface)
                            s6 = calc_interval_score(past)
                            s7 = calc_finish3f_score(past)

                            import re as _re
                            age_m   = _re.search(r"(\d+)", age_str)
                            age_val = int(age_m.group(1)) if age_m else 5
                            growth  = (5 - age_val) * 0.3

                            train_text = get_training_text(driver, horse_id)
                            st.session_state.horse_cache.setdefault("_all_past", []).append(past)
                            past_map[hname] = past

                            results.append({
                                "馬名":         hname,
                                "騎手":         jockey_name,
                                "スピード指数": round(s1, 2),
                                "能力指数":     round(s2, 2),
                                "EB指数":       round(s3, 2),
                                "Base指数":     round(s4, 2),
                                "騎手指数":     round(s5, 2),
                                "間隔スコア":   round(s6, 2),
                                "上がり3F":     round(s7, 2),
                                "調教内容":     train_text,
                                "_s1":s1,"_s2":s2,"_s3":s3,"_s4":s4,
                                "_s5":s5,"_s6":s6,"_s7":s7,
                                "_growth":growth,
                            })

                        progress.empty()
                        status.empty()

                        # mu_prior 更新
                        all_past = st.session_state.horse_cache.pop("_all_past", [])
                        if all_past:
                            st.session_state.horse_cache["mu_prior"] = update_mu_prior(all_past)

                        if not results:
                            st.warning(f"{race_name}：計算できる馬がありませんでした")
                            continue

                        # ── サンプル蓄積 ──
                        race_key = re.search(r"race_id=(\w+)", url)
                        race_key = race_key.group(1) if race_key else race_name
                        new_samples = accumulate_samples(results, race_key, past_map)
                        st.session_state.opt_samples.extend(new_samples)
                        st.session_state.opt_race_count += 1

                        # ── ウェイト最適化（10レース以上・5レース毎） ──
                        n_r = st.session_state.opt_race_count
                        smp = st.session_state.opt_samples
                        if (n_r >= MIN_RACES_FOR_OPT and
                                len(smp) >= MIN_SAMPLES_FOR_OPT and
                                n_r % 5 == 0):
                            with st.spinner("🔬 ウェイト最適化中..."):
                                ow = optimize_weights(smp)
                                st.session_state.opt_weights = ow
                                labels = "①②③④⑤⑥⑦"
                                st.info(
                                    f"✅ 最適化完了 ({n_r}R/{len(smp)}サンプル) "
                                    + " ".join(f"{labels[j]}{ow[j]:.2f}" for j in range(len(ow)))
                                )

                        # ── レース内正規化 + 総合指数計算 ──
                        keys  = ["_s1","_s2","_s3","_s4","_s5","_s6","_s7"]
                        norms = {
                            k: normalize_scores([r[k] for r in results])
                            for k in keys
                        }
                        for i, r in enumerate(results):
                            norm_scores = {
                                "s1": norms["_s1"][i], "s2": norms["_s2"][i],
                                "s3": norms["_s3"][i], "s4": norms["_s4"][i],
                                "s5": norms["_s5"][i], "s6": norms["_s6"][i],
                                "s7": norms["_s7"][i],
                            }
                            r["総合指数"] = round(calc_total(norm_scores, w, r["_growth"]), 2)
                            for k in keys + ["_growth"]:
                                r.pop(k, None)

                        result_df = ranked_df(
                            pd.DataFrame(results).sort_values("総合指数", ascending=False)
                        )

                        # ── 結果表示 ──
                        st.success(f"🏆 {race_name}　指数ランキング")
                        st.caption(w_desc)

                        show_cols = [c for c in [
                            "馬名","騎手","総合指数","スピード指数","能力指数",
                            "EB指数","Base指数","騎手指数","間隔スコア","上がり3F"
                        ] if c in result_df.columns]
                        st.dataframe(
                            result_df[show_cols],
                            use_container_width=True,
                            height=min(55 + len(result_df)*36, 680),
                        )

                        st.divider()
                        st.subheader("🎯 推奨買い目（指数ベース）")
                        recommend_tickets(result_df)

                        st.divider()
                        with st.spinner("🤖 Gemini AI で詳細分析中..."):
                            ai_text = call_gemini(
                                build_analysis_prompt(race_name, result_df),
                                GEMINI_API_KEY,
                            )
                            ai_data = parse_ai_response(ai_text)
                        render_ai_result(ai_data, ai_text)

                    except Exception as e:
                        import traceback
                        st.error(f"{race_name} 処理エラー: {str(e)[:300]}")
                        with st.expander("エラー詳細"):
                            st.code(traceback.format_exc())

st.divider()
st.caption("🤖 AI分析：Gemini 2.5 Flash  |  データ：netkeiba.com  |  7指数統合モデル × 競馬場別ウェイト")