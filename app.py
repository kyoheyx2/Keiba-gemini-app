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

# ===================== APIキー =====================
def get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return ""

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.5-flash"

# ===================== ページ設定 =====================
st.set_page_config(page_title="競馬指数アプリ", page_icon="🏇", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 16px; }
.block-container { padding: 3rem 0.75rem 2rem !important; max-width: 100% !important; }
h1 { font-size: 1.4rem !important; line-height: 1.3 !important; margin-top: 0.5rem !important; }
h2 { font-size: 1.2rem !important; }
h3 { font-size: 1.05rem !important; }
div.stButton > button {
    width: 100% !important; min-height: 3rem !important;
    font-size: 1rem !important; border-radius: 8px !important; margin-bottom: 0.5rem !important;
}
div.stButton > button[kind="primary"] { font-size: 1.1rem !important; min-height: 3.5rem !important; }
div[data-testid="stSelectbox"], div[data-testid="stDateInput"] { width: 100% !important; }
div[data-testid="stCheckbox"] { padding: 0.4rem 0 !important; }
div[data-testid="stCheckbox"] label { font-size: 1rem !important; min-height: 2.2rem !important; display: flex !important; align-items: center !important; }
div[data-testid="stCheckbox"] input[type="checkbox"] { width: 1.4rem !important; height: 1.4rem !important; margin-right: 0.6rem !important; }
div[data-testid="stTabs"] { overflow-x: auto !important; -webkit-overflow-scrolling: touch !important; }
button[data-testid="stTab"] { font-size: 0.95rem !important; padding: 0.5rem 0.8rem !important; white-space: nowrap !important; }
div[data-testid="stDataFrame"] { overflow-x: auto !important; -webkit-overflow-scrolling: touch !important; }
div[data-testid="stDataFrame"] table { font-size: 0.85rem !important; }
div[data-testid="stDataFrame"] th, div[data-testid="stDataFrame"] td { padding: 0.35rem 0.5rem !important; white-space: nowrap !important; }
@media (max-width: 600px) {
    h1 { font-size: 1.2rem !important; }
    div.stButton > button { min-height: 3.2rem !important; font-size: 0.95rem !important; }
    div[data-testid="stDataFrame"] table { font-size: 0.78rem !important; }
}
</style>
""", unsafe_allow_html=True)

st.title("🏇 競馬指数アプリ")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY が未設定です。ローカルは .streamlit/secrets.toml、Streamlit Cloud は Settings → Secrets に GEMINI_API_KEY を追加してください。")

selected_date = st.date_input("📅 分析したい日付", value=datetime.today().date())
kaisai_date = selected_date.strftime("%Y%m%d")

if 'driver'      not in st.session_state: st.session_state.driver      = None
if 'race_links'  not in st.session_state: st.session_state.race_links  = []
if 'horse_cache' not in st.session_state: st.session_state.horse_cache = {}


# ===================== 汎用 =====================
def make_unique_columns(cols):
    seen, new_cols = {}, []
    for col in cols:
        if col not in seen:
            seen[col] = 0; new_cols.append(col)
        else:
            seen[col] += 1; new_cols.append(f"{col}_{seen[col]}")
    return new_cols

def find_col(df, keywords):
    for kw in keywords:
        for c in df.columns:
            if kw in c: return c
    return None

def wait_for_page(driver, css_selector, timeout=10, fallback_sleep=2):
    try:
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)))
    except Exception:
        time.sleep(fallback_sleep)

def get_soup(driver):
    return BeautifulSoup(driver.page_source, "html.parser")

def ranked_df(df: pd.DataFrame) -> pd.DataFrame:
    """インデックスを1始まりにして '順位' という名前にする"""
    out = df.reset_index(drop=True).copy()
    out.index = range(1, len(out) + 1)
    out.index.name = "順位"
    return out


# ===================== Gemini呼び出し =====================
def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "⚠️ APIキーが未設定です"
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.HTTPError as e:
        return f"⚠️ APIエラー ({e.response.status_code}): {e.response.text[:200]}"
    except Exception as e:
        return f"⚠️ 通信エラー: {str(e)[:200]}"

def build_analysis_prompt(race_name, result_df):
    rows = [
        f"  {r['馬名']}: 総合{r['総合指数']} スピード{r['スピード']} クラス{r['クラス']} "
        f"距離{r['距離適性']} 馬場{r['馬場適性']} 脚質{r['脚質']} 調教{r['調教']} "
        f"騎手{r['騎手能力']} 騎手×脚質{r['騎手脚質相性']}"
        for _, r in result_df.iterrows()
    ]
    return f"""あなたは競馬の専門アナリストです。
以下は「{race_name}」の出走馬の指数データです。

{chr(10).join(rows)}

必ずJSON形式のみで出力してください。前置き・説明文・マークダウン不要。

{{
  "horses": [
    {{
      "馬名": "馬名",
      "勝率": 整数(0〜100),
      "連対率": 整数(0〜100),
      "複勝率": 整数(0〜100),
      "コメント": "50字以内のコメント"
    }}
  ],
  "レース総評": "100字以内"
}}

条件: 全馬の勝率合計=100、連対率・複勝率は勝率より高く設定。"""

def parse_ai_response(text):
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return None


# ===================== 過去走取得 =====================
def get_past_races(driver, horse_id):
    cache = st.session_state.horse_cache
    if horse_id in cache and cache[horse_id].get("past") is not None:
        return cache[horse_id]["past"]

    driver.get(f"https://db.netkeiba.com/horse/{horse_id}/")
    wait_for_page(driver, "table", timeout=8, fallback_sleep=2)
    soup = get_soup(driver)

    table = None
    for t in soup.find_all("table"):
        cls = " ".join(t.get("class", []))
        if "race_table" in cls or "db_h_race_results" in cls:
            table = t; break
    if table is None:
        all_t = soup.find_all("table")
        if all_t: table = max(all_t, key=lambda t: len(t.find_all("tr")))
    if table is None:
        cache.setdefault(horse_id, {})["past"] = None; return None

    try:
        df = pd.read_html(io.StringIO(str(table)))[0]
    except Exception:
        cache.setdefault(horse_id, {})["past"] = None; return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if str(c) != "nan"]).strip() for col in df.columns]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    df.columns = make_unique_columns(df.columns.tolist())

    order_col = next((c for c in df.columns if "着" in c and "順" in c), None)
    if order_col:
        df = df[df[order_col] != order_col].reset_index(drop=True)

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
    if table:
        try:
            df = pd.read_html(io.StringIO(str(table)))[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join([str(c) for c in col if str(c) != "nan"]).strip() for col in df.columns]
            else:
                df.columns = [str(c).strip() for c in df.columns]
            df.columns = make_unique_columns(df.columns.tolist())
            df = df.head(3)
            time_col = find_col(df, ["タイム", "時計"])
            if time_col:
                def parse_time(val):
                    val = str(val)
                    m = re.match(r'(\d+):(\d+\.\d+)', val)
                    if m: return int(m.group(1)) * 60 + float(m.group(2))
                    m = re.match(r'(\d+\.\d+)', val)
                    if m: return float(m.group(1))
                    return None
                df["_sec"] = df[time_col].apply(parse_time)
                mean_sec = df["_sec"].mean()
                if not pd.isna(mean_sec):
                    score = max((85 - mean_sec) * 0.5, 0)
        except Exception:
            pass

    cache.setdefault(horse_id, {})["train"] = score
    return score


# ===================== 騎手データ取得 =====================
def get_jockey_stats(driver, horse_id, past_df):
    """
    過去走から騎手IDを取得し、騎手の成績ページからスコアを算出。
    キャッシュは jockey_{jockey_id} キーで保存。
    """
    cache = st.session_state.horse_cache

    # 過去走テーブルから騎手名を取得
    jockey_col = find_col(past_df, ["騎手"])
    if jockey_col is None:
        return 5.0, 5.0  # データなし → 中央値

    # 直近の騎手名（今回の騎乗騎手と仮定）
    jockey_name = str(past_df[jockey_col].iloc[0]).strip()
    if not jockey_name or jockey_name in ("nan", ""):
        return 5.0, 5.0

    cache_key = f"jockey_{jockey_name}"
    if cache_key in cache:
        return cache[cache_key]["ability"], cache[cache_key]["compat"]

    # ---- 騎手検索ページでIDを取得 ----
    try:
        search_url = f"https://db.netkeiba.com/?pid=jockey_search&q={requests.utils.quote(jockey_name)}&old=y"
        driver.get(search_url)
        wait_for_page(driver, "table", timeout=6, fallback_sleep=1)
        soup = get_soup(driver)

        jockey_id = None
        for a in soup.find_all("a", href=True):
            m = re.search(r'/jockey/result/recent/(\w+)/', a["href"])
            if m:
                jockey_id = m.group(1)
                break

        if not jockey_id:
            cache[cache_key] = {"ability": 5.0, "compat": 5.0}
            return 5.0, 5.0

        # ---- 騎手成績ページ ----
        driver.get(f"https://db.netkeiba.com/jockey/result/recent/{jockey_id}/")
        wait_for_page(driver, "table", timeout=6, fallback_sleep=1)
        soup = get_soup(driver)

        tables = soup.find_all("table")
        ability_score = 5.0
        compat_score  = 5.0

        if tables:
            try:
                df = pd.read_html(io.StringIO(str(tables[0])))[0]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ["_".join([str(c) for c in col if str(c) != "nan"]).strip() for col in df.columns]
                else:
                    df.columns = [str(c).strip() for c in df.columns]
                df.columns = make_unique_columns(df.columns.tolist())

                win_col   = find_col(df, ["勝率"])
                place_col = find_col(df, ["連対率", "複勝率"])
                wins_col  = find_col(df, ["1着"])
                races_col = find_col(df, ["着数", "レース数", "出走"])

                # 勝率から騎手能力スコア（0〜10）
                if win_col:
                    win_rate = pd.to_numeric(
                        df[win_col].astype(str).str.replace("%","").str.strip(),
                        errors="coerce"
                    ).dropna()
                    if not win_rate.empty:
                        wr = win_rate.iloc[0]
                        # 勝率15%=10点、10%=7点、5%=4点を基準
                        ability_score = min(wr * 0.6 + 1.0, 10.0)

                # 連対率から騎手×脚質相性スコア
                if place_col:
                    place_rate = pd.to_numeric(
                        df[place_col].astype(str).str.replace("%","").str.strip(),
                        errors="coerce"
                    ).dropna()
                    if not place_rate.empty:
                        pr = place_rate.iloc[0]
                        compat_score = min(pr * 0.25 + 1.0, 10.0)

            except Exception:
                pass

        cache[cache_key] = {"ability": ability_score, "compat": compat_score}
        return ability_score, compat_score

    except Exception:
        cache[cache_key] = {"ability": 5.0, "compat": 5.0}
        return 5.0, 5.0


# ===================== 指数計算 =====================
def class_score_val(val):
    if not isinstance(val, str): return 0
    if "G1" in val: return 10
    if "G2" in val: return 8
    if "G3" in val: return 6
    if "OP" in val or "オープン" in val: return 5
    if "3勝" in val or "1600万" in val: return 4
    if "2勝" in val or "1000万" in val: return 3
    if "1勝" in val or "500万" in val: return 2
    if "未勝利" in val or "新馬" in val: return 1
    return 0

WEIGHTS = [1.0, 0.75, 0.55, 0.4, 0.3]

def weighted_mean(series, n):
    w = WEIGHTS[:n]
    tw = sum(w)
    return sum(series.iloc[i] * w[i] for i in range(n)) / tw if tw > 0 else 0.0

def calc_speed_index(past_df):
    df = past_df.copy().reset_index(drop=True)
    n = len(df)
    ti_col  = find_col(df, ["タイム指数"])
    ag_col  = find_col(df, ["上り", "上がり", "上3F"])
    mg_col  = find_col(df, ["着差"])
    ord_col = find_col(df, ["着順"])

    df["_ti"]  = pd.to_numeric(df[ti_col],  errors="coerce").fillna(0)    if ti_col  else pd.Series([0.0]*n)
    df["_ag"]  = pd.to_numeric(df[ag_col],  errors="coerce").fillna(36.5) if ag_col  else pd.Series([36.5]*n)
    df["_mg"]  = pd.to_numeric(df[mg_col],  errors="coerce").fillna(1.0)  if mg_col  else pd.Series([1.0]*n)
    df["_ord"] = pd.to_numeric(df[ord_col], errors="coerce").fillna(9.0)  if ord_col else pd.Series([9.0]*n)

    df["_raw"] = (
        df["_ti"] * 1.2 +
        (36.5 - df["_ag"]).clip(lower=-5) * 2.5 +
        (11 - df["_ord"]).clip(lower=0) * 1.5 +
        (1.5 - df["_mg"].clip(lower=-1.0)) * 3.0
    )
    return weighted_mean(df["_raw"], n)

def calc_class_score(past_df):
    df = past_df.copy().reset_index(drop=True)
    race_col = find_col(df, ["レース名", "競走名", "条件", "クラス"])
    if not race_col: return 0
    scores = df[race_col].astype(str).apply(class_score_val)
    return weighted_mean(scores, len(scores))

def calc_distance_score(past_df, target_dist):
    df = past_df.copy()
    dist_col = find_col(df, ["距離"])
    ord_col  = find_col(df, ["着順"])
    if not dist_col: return 0
    df["_d"]   = df[dist_col].astype(str).str.extract(r'(\d{3,4})').astype(float)
    df["_ord"] = pd.to_numeric(df[ord_col], errors="coerce").fillna(9.0) if ord_col else 9.0
    df["_diff"]  = (df["_d"] - target_dist).abs()
    df["_score"] = (3000 - df["_diff"]) / 100
    df["_bonus"] = ((df["_diff"] <= 200) & (df["_ord"] <= 3)).astype(float) * 3.0
    return max((df["_score"] + df["_bonus"]).mean(), 0)

def calc_style_score(past_df):
    df = past_df.copy()
    pass_col = find_col(df, ["通過", "コーナー", "通過順"])
    if not pass_col: return 5.0
    df["_pos"] = df[pass_col].astype(str).str.extract(r'(\d+)').astype(float)
    mean_pos = df["_pos"].mean()
    if pd.isna(mean_pos): return 5.0
    if 2 <= mean_pos <= 4:  return 10.0
    if mean_pos < 2:        return 7.0
    if mean_pos <= 7:       return max(10 - (mean_pos - 4) * 1.2, 3.0)
    return 3.0

def calc_surface_score(past_df, surface="芝"):
    df = past_df.copy()
    dist_col = find_col(df, ["距離", "コース"])
    ord_col  = find_col(df, ["着順"])
    if not dist_col: return 5.0
    df["_match"] = df[dist_col].astype(str).str.contains(surface, na=False)
    df["_ord"]   = pd.to_numeric(df[ord_col], errors="coerce").fillna(9.0) if ord_col else 9.0
    match_df = df[df["_match"]]
    if match_df.empty: return 3.0
    return max((10 - match_df["_ord"].mean()) * 1.2, 0)


# ===================== 馬ID取得 =====================
def extract_horse_id(href):
    if not href: return None
    m = re.search(r'/horse/(\d{10,})', href)
    if m: return m.group(1)
    m = re.search(r"goHorse\('(\d+)'\)", href)
    if m: return m.group(1)
    return None

def get_horse_ids_from_page(driver, soup):
    horse_info, seen_ids = [], set()
    for a in soup.find_all("a", href=True):
        hid = extract_horse_id(a["href"])
        if hid and hid not in seen_ids:
            name = a.get_text(strip=True)
            if name and len(name) >= 2:
                horse_info.append((name, hid)); seen_ids.add(hid)
    if horse_info: return horse_info
    try:
        for a in driver.find_elements(By.CSS_SELECTOR, "a[href*='/horse/']"):
            href = a.get_attribute("href") or ""
            hid = extract_horse_id(href)
            if hid and hid not in seen_ids:
                name = a.text.strip()
                if name and len(name) >= 2:
                    horse_info.append((name, hid)); seen_ids.add(hid)
    except Exception:
        pass
    return horse_info


# ===================== レース一覧取得 =====================
if st.button(f"🔍 {selected_date} の全レース一覧を取得"):
    with st.spinner("取得中..."):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        try:
            import shutil, subprocess, os

            # Streamlit Cloud上のChromiumパスを確実に特定する
            chromium_path = (
                shutil.which("chromium-browser") or
                shutil.which("chromium") or
                shutil.which("google-chrome") or
                shutil.which("google-chrome-stable")
            )
            chromedriver_path = (
                shutil.which("chromedriver") or
                "/usr/bin/chromedriver"
            )

            # chromiumが見つかった場合はそのパスをセット
            if chromium_path:
                options.binary_location = chromium_path

            # ChromeDriverのパスが実在するか確認
            if os.path.exists(chromedriver_path):
                service = Service(chromedriver_path)
            else:
                # ローカルWindows環境ではWebDriverManagerを使用
                service = Service(ChromeDriverManager().install())

            driver  = webdriver.Chrome(service=service, options=options)
            st.session_state.driver      = driver
            st.session_state.horse_cache = {}

            driver.get(f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={kaisai_date}")
            wait_for_page(driver, "a[href*='shutuba.html']", timeout=10, fallback_sleep=3)

            race_links, seen_urls = [], set()
            for elem in driver.find_elements(By.TAG_NAME, "a"):
                href = elem.get_attribute("href") or ""
                text = elem.text.strip()
                if "shutuba.html?race_id=" in href and href not in seen_urls and len(text) > 3:
                    race_links.append((text, href)); seen_urls.add(href)

            st.session_state.race_links = race_links
            st.success(f"✅ {len(race_links)}レース取得完了！")
        except Exception as e:
            st.error(f"取得エラー: {e}")


# ===================== レース選択 UI =====================
if st.session_state.race_links:
    st.subheader("🏇 レースを選択してください")

    grouped  = defaultdict(list)
    place_map = {"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京",
                 "06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}

    for name, url in st.session_state.race_links:
        match = re.search(r'race_id=(\d{4})(\d{2})\d+', url)
        venue = place_map.get(match.group(2) if match else "", "その他")
        grouped[venue].append((name, url))

    venues = [v for v in ["東京","京都","阪神","中山","新潟","小倉","福島","中京","札幌","函館","その他"] if v in grouped]
    tabs   = st.tabs(venues)

    selected_races = []
    for tab, venue in zip(tabs, venues):
        with tab:
            st.markdown(f"**🏟️ {venue}**")
            for name, url in grouped[venue]:
                if st.checkbox(name, key=f"check_{name}"):
                    selected_races.append((name, url))

    if selected_races:
        st.info(f"✅ {len(selected_races)}レース選択中")

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
                        for kw in ["shutuba_table","Shutuba_Table","race_table_01"]:
                            table = soup.find("table", class_=lambda x: x and kw in " ".join(x) if x else False)
                            if table: break
                        if table is None:
                            all_t = soup.find_all("table")
                            if all_t: table = max(all_t, key=lambda t: len(t.find_all("tr")))
                        if table is None:
                            st.error(f"{race_name}：テーブルが見つかりませんでした"); continue

                        horse_info = get_horse_ids_from_page(driver, soup)
                        if not horse_info:
                            st.error(f"{race_name}：馬IDが取得できませんでした"); continue
                        st.info(f"→ {len(horse_info)}頭取得")

                        # レース距離・馬場
                        race_distance, race_surface = 0, "芝"
                        for kw in ["RaceData01","race_data","mainrace_data"]:
                            tag = soup.find(class_=kw)
                            if tag:
                                txt = tag.get_text()
                                m = re.search(r'(\d{3,4})m', txt)
                                if m: race_distance = int(m.group(1))
                                if "ダ" in txt or "ダート" in txt: race_surface = "ダ"
                                break
                        if race_distance == 0:
                            m = re.search(r'(\d{3,4})m', soup.get_text())
                            if m: race_distance = int(m.group(1))

                        # 斤量・性齢・騎手マップ
                        burden_map, age_map, jockey_map = {}, {}, {}
                        try:
                            df_s = pd.read_html(io.StringIO(str(table)))[0]
                            if isinstance(df_s.columns, pd.MultiIndex):
                                df_s.columns = ["_".join([str(c) for c in col if str(c) != "nan"]).strip() for col in df_s.columns]
                            else:
                                df_s.columns = [str(c).strip() for c in df_s.columns]
                            df_s.columns = make_unique_columns(df_s.columns.tolist())
                            burden_col = find_col(df_s, ["斤量"])
                            age_col    = find_col(df_s, ["性齢"])
                            name_col   = find_col(df_s, ["馬名"])
                            jockey_col = find_col(df_s, ["騎手"])
                            if name_col:
                                for _, row in df_s.iterrows():
                                    hn = str(row[name_col]).strip()
                                    if burden_col: burden_map[hn] = pd.to_numeric(row[burden_col], errors="coerce")
                                    if age_col:    age_map[hn]    = str(row[age_col])
                                    if jockey_col: jockey_map[hn] = str(row[jockey_col]).strip()
                        except Exception:
                            pass

                        # 各馬の指数計算
                        results  = []
                        progress = st.progress(0)
                        status   = st.empty()

                        for idx, (hname, horse_id) in enumerate(horse_info):
                            progress.progress((idx + 1) / len(horse_info))
                            status.caption(f"処理中: {hname} ({idx+1}/{len(horse_info)})")

                            past = get_past_races(driver, horse_id)
                            if past is None or past.empty: continue

                            speed      = calc_speed_index(past)
                            cls        = calc_class_score(past)
                            dist_score = calc_distance_score(past, race_distance)
                            style      = calc_style_score(past)
                            surface    = calc_surface_score(past, race_surface)
                            train      = calc_training_score(driver, horse_id)

                            # 騎手能力・騎手×脚質相性
                            jockey_ability, jockey_compat = get_jockey_stats(driver, horse_id, past)

                            burden     = burden_map.get(hname, 55.0)
                            burden_adj = float((55 - burden) * 4.0) if pd.notna(burden) else 0.0
                            age_str    = age_map.get(hname, "")
                            age_m      = re.search(r'(\d+)', age_str)
                            age_val    = int(age_m.group(1)) if age_m else 5
                            growth_adj = float((5 - age_val) * 4.0)

                            total = (
                                speed          * 0.35 +
                                cls            * 0.13 +
                                dist_score     * 0.13 +
                                surface        * 0.08 +
                                style          * 0.07 +
                                train          * 0.07 +
                                jockey_ability * 0.10 +  # 騎手能力
                                jockey_compat  * 0.05 +  # 騎手×脚質相性
                                burden_adj     * 0.01 +
                                growth_adj     * 0.01
                            )
                            results.append({
                                "馬名":       hname,
                                "総合指数":   round(total, 2),
                                "スピード":   round(speed, 2),
                                "クラス":     round(cls, 2),
                                "距離適性":   round(dist_score, 2),
                                "馬場適性":   round(surface, 2),
                                "脚質":       round(style, 2),
                                "調教":       round(train, 2),
                                "騎手能力":   round(jockey_ability, 2),
                                "騎手脚質相性": round(jockey_compat, 2),
                                "斤量補正":   round(burden_adj, 2),
                                "成長補正":   round(growth_adj, 2),
                            })

                        progress.empty()
                        status.empty()

                        if not results:
                            st.warning(f"{race_name}：指数計算できる馬がいませんでした"); continue

                        # ★ sort → 1始まりインデックス
                        result_df = ranked_df(
                            pd.DataFrame(results).sort_values("総合指数", ascending=False)
                        )

                        st.success(f"🏆 {race_name} 指数ランキング")
                        # ★ expander なし・1テーブルのみ表示
                        st.dataframe(
                            result_df[["馬名","総合指数","スピード","クラス","距離適性","馬場適性","脚質","騎手能力","騎手脚質相性"]],
                            use_container_width=True,
                            height=min(50 + len(result_df) * 35, 600),
                        )

                        # AI分析
                        st.divider()
                        with st.spinner("🤖 Gemini 2.5 Flash でAI分析中..."):
                            ai_text = call_gemini(build_analysis_prompt(race_name, result_df))
                            ai_data = parse_ai_response(ai_text)

                        if ai_data and "horses" in ai_data:
                            st.subheader("🤖 AI分析（Gemini 2.5 Flash）")
                            if "レース総評" in ai_data:
                                st.info(f"📝 {ai_data['レース総評']}")

                            ai_df = ranked_df(
                                pd.DataFrame([{
                                    "馬名":      h.get("馬名",""),
                                    "勝率(%)":   h.get("勝率", 0),
                                    "連対率(%)": h.get("連対率", 0),
                                    "複勝率(%)": h.get("複勝率", 0),
                                    "AIコメント": h.get("コメント",""),
                                } for h in ai_data["horses"]]).sort_values("勝率(%)", ascending=False)
                            )

                            st.markdown("**📈 勝率・連対率・複勝率**")
                            st.dataframe(
                                ai_df[["馬名","勝率(%)","連対率(%)","複勝率(%)"]],
                                use_container_width=True,
                                height=min(50 + len(ai_df) * 35, 600),
                            )
                            with st.expander("💬 AIコメントを表示"):
                                for rank, row in ai_df.iterrows():
                                    st.markdown(f"**{rank}位 {row['馬名']}**：{row['AIコメント']}")
                        else:
                            st.subheader("🤖 AI分析（Gemini 2.5 Flash）")
                            st.text(ai_text)

                    except Exception as e:
                        import traceback
                        st.error(f"{race_name} 処理エラー: {str(e)[:300]}")
                        with st.expander("エラー詳細"):
                            st.code(traceback.format_exc(), language="python")

st.caption("🤖 AI分析：Gemini 2.5 Flash 使用")