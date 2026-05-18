import streamlit as st
import pandas as pd
import io
import json
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
from datetime import datetime
from collections import defaultdict
import re
from bs4 import BeautifulSoup

# ===================== Gemini APIキー（固定値） =====================
GEMINI_API_KEY = "AIzaSyCnlvx_PtjPnE9PAdZJnJFHl5yourt5qRs"
GEMINI_MODEL = "gemini-1.5-pro-latest"

# ===================== Streamlit 基本設定 =====================
st.set_page_config(page_title="俺の自動競馬指数アプリ（Gemini専用）", layout="wide")
st.title("🚀 競馬場タブ + 本格独自指数アプリ（Gemini専用）")

selected_date = st.date_input("📅 分析したい日付", value=datetime.today().date())
kaisai_date = selected_date.strftime("%Y%m%d")

if 'driver' not in st.session_state:
    st.session_state.driver = None
if 'race_links' not in st.session_state:
    st.session_state.race_links = []
if 'horse_cache' not in st.session_state:
    st.session_state.horse_cache = {}

# ===================== 汎用関数 =====================
def make_unique_columns(cols):
    seen = {}
    new_cols = []
    for col in cols:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    return new_cols

def find_col(df, keywords):
    for kw in keywords:
        for c in df.columns:
            if kw in c:
                return c
    return None

def wait_for_page(driver, css_selector, timeout=10, fallback_sleep=2):
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
        )
    except Exception:
        time.sleep(fallback_sleep)

def get_soup(driver):
    return BeautifulSoup(driver.page_source, "html.parser")

# ===================== Gemini呼び出し =====================
def call_gemini(prompt: str) -> str:
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": "AIzaSyCnlvx_PtjPnE9PAdZJnJFHl5yourt5qRs"
            },
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ]
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"⚠️ Gemini APIエラー: {str(e)[:200]}"

def build_analysis_prompt(race_name: str, result_df: pd.DataFrame) -> str:
    rows = []
    for _, r in result_df.iterrows():
        rows.append(
            f"  {r['馬名']}: 総合{r['総合指数']} スピード{r['スピード']} クラス{r['クラス']} "
            f"距離適性{r['距離適性']} 脚質{r['脚質']} 調教{r['調教']} "
            f"斤量補正{r['斤量補正']} 成長補正{r['成長補正']}"
        )
    horses_text = "\n".join(rows)

    return f"""あなたは競馬の専門アナリストです。
以下は「{race_name}」の出走馬の各種指数データです。

{horses_text}

このデータをもとに、以下を**必ずJSON形式のみ**で出力してください。
前置き・説明文・マークダウンは一切不要です。JSONだけ出力してください。

{{
  "horses": [
    {{
      "馬名": "馬名",
      "勝率": 数値(0〜100の整数),
      "連対率": 数値(0〜100の整数),
      "複勝率": 数値(0〜100の整数),
      "コメント": "簡潔な分析コメント（50字以内）"
    }}
  ],
  "レース総評": "レース全体の見どころ（100字以内）"
}}

条件:
- 全馬の勝率の合計が100になるよう調整してください
- 連対率・複勝率は勝率より高く、現実的な範囲で設定してください
- データが少ない馬は控えめに評価してください
"""

def parse_ai_response(text: str) -> dict | None:
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None
# ===================== 過去走取得（キャッシュ付き） =====================
def get_past_races(driver, horse_id):
    cache = st.session_state.horse_cache
    if horse_id in cache and cache[horse_id].get("past") is not None:
        return cache[horse_id]["past"]

    driver.get(f"https://db.netkeiba.com/horse/{horse_id}/")
    wait_for_page(driver, "table", timeout=8, fallback_sleep=2)

    soup = get_soup(driver)

    # 競走成績テーブルを確実に拾う
    tables = soup.find_all("table", summary=lambda x: x and ("競走成績" in x or "戦績" in x))
    if not tables:
        tables = soup.find_all("table", class_=lambda x: x and "race_results" in str(x))
    if not tables:
        tables = soup.find_all("table", class_=lambda x: x and "db_h_race_results" in str(x))

    if not tables:
        cache.setdefault(horse_id, {})["past"] = None
        return None

    try:
        df = pd.read_html(io.StringIO(str(tables[0])))[0]
    except Exception:
        cache.setdefault(horse_id, {})["past"] = None
        return None

    df.columns = make_unique_columns([str(c).strip() for c in df.columns])
    result = df.head(5)
    cache.setdefault(horse_id, {})["past"] = result
    return result


# ===================== 調教スコア =====================
def calc_training_score(driver, horse_id):
    cache = st.session_state.horse_cache
    if horse_id in cache and "train" in cache[horse_id]:
        return cache[horse_id]["train"]

    driver.get(f"https://db.netkeiba.com/horse/{horse_id}/training/")
    wait_for_page(driver, "table", timeout=6, fallback_sleep=1)

    soup = get_soup(driver)
    table = soup.find("table")
    score = 0

    if table is not None:
        try:
            df = pd.read_html(io.StringIO(str(table)))[0]
            df.columns = make_unique_columns([str(c).strip() for c in df.columns])
            df = df.head(3)

            time_col = find_col(df, ["タイム", "時計"])
            if time_col:
                def parse_time(val):
                    val = str(val)
                    m = re.match(r'(\d+):(\d+\.\d+)', val)
                    if m:
                        return int(m.group(1)) * 60 + float(m.group(2))
                    m = re.match(r'(\d+\.\d+)', val)
                    if m:
                        return float(m.group(1))
                    return None

                df["_sec"] = df[time_col].apply(parse_time)
                mean_sec = df["_sec"].mean()
                if not pd.isna(mean_sec):
                    score = max((85 - mean_sec) * 0.5, 0)
        except Exception:
            pass

    cache.setdefault(horse_id, {})["train"] = score
    return score


# ===================== 馬ID取得 =====================
def extract_horse_id(href_or_onclick):
    if not href_or_onclick:
        return None
    m = re.search(r'/horse/(\d{10,})', href_or_onclick)
    if m:
        return m.group(1)
    m = re.search(r"goHorse\('(\d+)'\)", href_or_onclick)
    if m:
        return m.group(1)
    return None


def get_horse_ids_from_page(driver, soup):
    horse_info = []
    seen_ids = set()

    for a in soup.find_all("a", href=True):
        horse_id = extract_horse_id(a["href"])
        if horse_id and horse_id not in seen_ids:
            name = a.get_text(strip=True)
            if name and len(name) >= 2:
                horse_info.append((name, horse_id))
                seen_ids.add(horse_id)

    return horse_info


# ===================== 指数計算関数 =====================
def class_score(val):
    if not isinstance(val, str):
        return 0
    if "G1" in val: return 10
    if "G2" in val: return 8
    if "G3" in val: return 6
    if "OP" in val or "オープン" in val: return 5
    if "3勝" in val or "1600万" in val: return 4
    if "2勝" in val or "1000万" in val: return 3
    if "1勝" in val or "500万" in val: return 2
    if "未勝利" in val or "新馬" in val: return 1
    return 0


def calc_speed_index(past_df):
    df = past_df.copy()
    time_idx_col = find_col(df, ["タイム指数"])
    agari_col    = find_col(df, ["上り", "上がり", "上3F"])
    margin_col   = find_col(df, ["着差"])

    df["_ti"]     = pd.to_numeric(df[time_idx_col], errors="coerce") if time_idx_col else 0
    df["_agari"]  = pd.to_numeric(df[agari_col], errors="coerce")    if agari_col    else 36.5
    df["_margin"] = pd.to_numeric(df[margin_col], errors="coerce")   if margin_col   else 1.0

    df["speed"] = (
        df["_ti"].fillna(0) * 1.2 +
        (36.5 - df["_agari"].fillna(36.5)) * 2.0 +
        (1.0 - df["_margin"].fillna(1.0)) * 4.0
    )
    return df["speed"].mean()


def calc_class_score_from_df(past_df):
    df = past_df.copy()
    race_col = find_col(df, ["レース名", "競走名", "条件", "クラス"])
    if not race_col:
        return 0
    return df[race_col].astype(str).apply(class_score).mean()


def calc_distance_score(past_df, target_distance):
    df = past_df.copy()
    dist_col = find_col(df, ["距離"])
    if not dist_col:
        return 0
    df["_dist"] = df[dist_col].astype(str).str.extract(r'(\d{3,4})').astype(float)
    diff = (df["_dist"] - target_distance).abs().mean()
    return max((3000 - diff) / 100, 0)


def calc_style_score(past_df):
    df = past_df.copy()
    pass_col = find_col(df, ["通過", "コーナー", "通過順"])
    if not pass_col:
        return 0
    df["_pos"] = df[pass_col].astype(str).str.extract(r'(\d+)').astype(float)
    mean_pos = df["_pos"].mean()
    if pd.isna(mean_pos):
        return 0
    return max((10 - mean_pos) * 1.5, 0)


# ===================== レース一覧取得 =====================
if st.button(f"🔍 {selected_date} の全レース一覧を取得"):
    with st.spinner("取得中..."):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )

        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            st.session_state.driver = driver
            st.session_state.horse_cache = {}

            driver.get(f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={kaisai_date}")
            wait_for_page(driver, "a[href*='shutuba.html']", timeout=10, fallback_sleep=3)

            race_links = []
            seen_urls = set()
            for elem in driver.find_elements(By.TAG_NAME, "a"):
                href = elem.get_attribute("href") or ""
                text = elem.text.strip()
                if "shutuba.html?race_id=" in href and href not in seen_urls:
                    if text and len(text) > 3:
                        race_links.append((text, href))
                        seen_urls.add(href)

            st.session_state.race_links = race_links
            st.success(f"✅ {len(race_links)}レース取得完了！")

        except Exception as e:
            st.error(f"取得エラー: {e}")
# ===================== レース選択 UI =====================
if st.session_state.race_links:
    st.subheader("🏇 分析したいレースにチェックを入れてください")

    grouped = defaultdict(list)
    place_map = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
        "05": "東京", "06": "中山", "07": "中京", "08": "京都",
        "09": "阪神", "10": "小倉"
    }

    for name, url in st.session_state.race_links:
        match = re.search(r'race_id=(\d{4})(\d{2})\d+', url)
        venue = place_map.get(match.group(2) if match else "", "その他")
        grouped[venue].append((name, url))

    venues = [v for v in ["東京","京都","阪神","中山","新潟","小倉","福島","中京","札幌","函館","その他"] if v in grouped]
    tabs = st.tabs(venues)

    selected_races = []
    for tab, venue in zip(tabs, venues):
        with tab:
            st.markdown(f"### 🏟️ **{venue}**")
            for name, url in grouped[venue]:
                if st.checkbox(name, key=f"check_{name}"):
                    selected_races.append((name, url))

    # ===================== 指数計算 =====================
    if st.button("📊 選択したレースの指数を計算する", type="primary"):
        if not selected_races:
            st.warning("レースを選択してください")
        else:
            driver = st.session_state.driver

            for race_name, url in selected_races:
                with st.spinner(f"{race_name} 処理中..."):
                    try:
                        driver.get(url)
                        wait_for_page(driver, "table", timeout=10, fallback_sleep=3)
                        soup = get_soup(driver)

                        # 出馬表テーブル
                        table = None
                        for cls_kw in ["shutuba_table", "Shutuba_Table", "race_table_01"]:
                            table = soup.find("table", class_=lambda x: x and cls_kw in " ".join(x) if x else False)
                            if table:
                                break
                        if table is None:
                            all_tables = soup.find_all("table")
                            if all_tables:
                                table = max(all_tables, key=lambda t: len(t.find_all("tr")))
                        if table is None:
                            st.error(f"{race_name} で出馬表テーブルが見つかりませんでした")
                            continue

                        # 馬ID取得
                        horse_info = get_horse_ids_from_page(driver, soup)
                        if not horse_info:
                            st.error(f"{race_name} で馬IDが取得できませんでした")
                            continue
                        st.info(f"  → {len(horse_info)}頭の馬IDを取得しました")

                        # 距離取得
                        race_distance = 0
                        for cls_kw in ["RaceData01", "race_data", "mainrace_data"]:
                            tag = soup.find(class_=cls_kw)
                            if tag:
                                m = re.search(r'(\d{3,4})m', tag.get_text())
                                if m:
                                    race_distance = int(m.group(1))
                                    break

                        # 斤量・性齢
                        burden_map, age_map = {}, {}
                        try:
                            df_shutuba = pd.read_html(io.StringIO(str(table)))[0]
                            df_shutuba.columns = make_unique_columns([str(c).strip() for c in df_shutuba.columns])
                            burden_col = find_col(df_shutuba, ["斤量"])
                            age_col    = find_col(df_shutuba, ["性齢"])
                            name_col   = find_col(df_shutuba, ["馬名"])
                            if name_col:
                                for _, row in df_shutuba.iterrows():
                                    hn = str(row[name_col]).strip()
                                    if burden_col:
                                        burden_map[hn] = pd.to_numeric(row[burden_col], errors="coerce")
                                    if age_col:
                                        age_map[hn] = str(row[age_col])
                        except Exception:
                            pass

                        # ===================== 指数計算本体 =====================
                        results = []
                        progress = st.progress(0)
                        status = st.empty()

                        for idx, (hname, horse_id) in enumerate(horse_info):
                            progress.progress((idx + 1) / len(horse_info))
                            status.caption(f"処理中: {hname} ({idx+1}/{len(horse_info)})")

                            past = get_past_races(driver, horse_id)
                            if past is None or past.empty:
                                continue

                            speed      = calc_speed_index(past)
                            cls        = calc_class_score_from_df(past)
                            dist_score = calc_distance_score(past, race_distance)
                            style      = calc_style_score(past)
                            train      = calc_training_score(driver, horse_id)

                            burden     = burden_map.get(hname, 55.0)
                            burden_adj = float((55 - burden) * 4.0) if pd.notna(burden) else 0.0
                            age_str    = age_map.get(hname, "")
                            age_m      = re.search(r'(\d+)', age_str)
                            age_val    = int(age_m.group(1)) if age_m else 5
                            growth_adj = float((5 - age_val) * 4.0)

                            total = (
                                speed      * 0.45 +
                                cls        * 0.15 +
                                dist_score * 0.15 +
                                style      * 0.10 +
                                train      * 0.10 +
                                burden_adj * 0.03 +
                                growth_adj * 0.02
                            )

                            results.append([
                                hname, round(total, 2), round(speed, 2), round(cls, 2),
                                round(dist_score, 2), round(style, 2), round(train, 2),
                                round(burden_adj, 2), round(growth_adj, 2)
                            ])

                        progress.empty()
                        status.empty()

                        if not results:
                            st.warning(f"{race_name}：指数計算できる馬がいませんでした")
                            continue

                        result_df = pd.DataFrame(
                            results,
                            columns=["馬名","総合指数","スピード","クラス","距離適性","脚質","調教","斤量補正","成長補正"]
                        ).sort_values("総合指数", ascending=False).reset_index(drop=True)

                        st.success(f"🏆 {race_name} の総合指数ランキング")
                        st.dataframe(result_df, use_container_width=True, height=400)

                        # ===================== Gemini AI分析 =====================
                        st.divider()
                        with st.spinner("🤖 Gemini でAI分析中..."):
                            prompt   = build_analysis_prompt(race_name, result_df)
                            ai_text  = call_gemini(prompt)
                            ai_data  = parse_ai_response(ai_text)

                        if ai_data and "horses" in ai_data:
                            st.subheader("🤖 Gemini AI分析結果")

                            if "レース総評" in ai_data:
                                st.info(f"📝 {ai_data['レース総評']}")

                            ai_rows = []
                            for h in ai_data["horses"]:
                                ai_rows.append({
                                    "馬名":   h.get("馬名", ""),
                                    "勝率(%)":  h.get("勝率", 0),
                                    "連対率(%)": h.get("連対率", 0),
                                    "複勝率(%)": h.get("複勝率", 0),
                                    "AIコメント": h.get("コメント", ""),
                                })

                            ai_df = pd.DataFrame(ai_rows).sort_values("勝率(%)", ascending=False).reset_index(drop=True)
                            st.dataframe(
                                ai_df.style.background_gradient(subset=["勝率(%)","連対率(%)","複勝率(%)"], cmap="YlOrRd"),
                                use_container_width=True,
                                height=400,
                            )
                        else:
                            st.subheader("🤖 Gemini AI分析結果（JSON解析失敗）")
                            st.text(ai_text)

                    except Exception as e:
                        import traceback
                        st.error(f"{race_name} 処理エラー: {str(e)[:300]}")
                        st.code(traceback.format_exc(), language="python")

st.caption("※ チェック選択 → 下のボタンで計算｜Gemini専用AI分析")