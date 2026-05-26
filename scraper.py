"""
scraper.py — netkeiba スクレイピング
"""
import io, re, time
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from config import PLACE_MAP
from utils import make_unique_columns, find_col


# ── ドライバー ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_driver():
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=opts
    )


def wait_for_page(driver, css: str, timeout=10, fallback=2):
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css))
        )
    except Exception:
        time.sleep(fallback)


def get_soup(driver) -> BeautifulSoup:
    return BeautifulSoup(driver.page_source, "html.parser")


# ── レース一覧取得 ──────────────────────────────────────────────
def fetch_race_list(driver, kaisai_date: str) -> list[tuple[str, str]]:
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={kaisai_date}"
    driver.get(url)
    wait_for_page(driver, "a[href*='shutuba.html']", timeout=12, fallback=3)
    links, seen = [], set()
    for a in driver.find_elements(By.TAG_NAME, "a"):
        href = a.get_attribute("href") or ""
        text = a.text.strip()
        if "shutuba.html?race_id=" in href and href not in seen and len(text) > 2:
            links.append((text, href))
            seen.add(href)
    return links


# ── 馬ID取得 ────────────────────────────────────────────────────
def extract_horse_id(href: str) -> str | None:
    if not href:
        return None
    m = re.search(r"/horse/(\d{10,})", href)
    if m:
        return m.group(1)
    m = re.search(r"goHorse\('(\d+)'\)", href)
    return m.group(1) if m else None


def get_horse_ids_from_page(driver, soup: BeautifulSoup) -> list[tuple[str, str]]:
    info, seen = [], set()
    for a in soup.find_all("a", href=True):
        hid = extract_horse_id(a["href"])
        if hid and hid not in seen:
            name = a.get_text(strip=True)
            if name and len(name) >= 2:
                info.append((name, hid))
                seen.add(hid)
    if info:
        return info
    for a in driver.find_elements(By.CSS_SELECTOR, "a[href*='/horse/']"):
        href = a.get_attribute("href") or ""
        hid = extract_horse_id(href)
        if hid and hid not in seen:
            name = a.text.strip()
            if name and len(name) >= 2:
                info.append((name, hid))
                seen.add(hid)
    return info


# ── 過去走取得 ──────────────────────────────────────────────────
def get_past_races(driver, horse_id: str) -> pd.DataFrame | None:
    cache = st.session_state.horse_cache
    cached = cache.get(horse_id, {}).get("past")
    if cached is not None:
        return cached

    driver.get(f"https://db.netkeiba.com/horse/{horse_id}/")
    wait_for_page(driver, "table", timeout=8, fallback=2)
    soup = get_soup(driver)

    table = None
    for t in soup.find_all("table"):
        cls = " ".join(t.get("class", []))
        if "race_table" in cls or "db_h_race_results" in cls:
            table = t
            break
    if table is None:
        all_t = soup.find_all("table")
        if all_t:
            table = max(all_t, key=lambda t: len(t.find_all("tr")))
    if table is None:
        cache.setdefault(horse_id, {})["past"] = None
        return None

    try:
        df = pd.read_html(io.StringIO(str(table)))[0]
    except Exception:
        cache.setdefault(horse_id, {})["past"] = None
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(c) for c in col if str(c) != "nan").strip()
            for col in df.columns
        ]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    df.columns = make_unique_columns(df.columns.tolist())

    order_col = next((c for c in df.columns if "着" in c and "順" in c), None)
    if order_col:
        df = df[df[order_col] != order_col].reset_index(drop=True)

    result = df.head(5)
    cache.setdefault(horse_id, {})["past"] = result
    if "debug_cols" not in cache:
        cache["debug_cols"] = result.columns.tolist()
    return result


# ── 調教テキスト取得 ────────────────────────────────────────────
def get_training_text(driver, horse_id: str) -> str:
    cache = st.session_state.horse_cache
    if horse_id in cache and "train_text" in cache[horse_id]:
        return cache[horse_id]["train_text"]
    text = ""
    try:
        driver.get(f"https://db.netkeiba.com/horse/{horse_id}/training/")
        wait_for_page(driver, "table", timeout=6, fallback=1)
        soup = get_soup(driver)
        tbl = soup.find("table")
        if tbl:
            df = pd.read_html(io.StringIO(str(tbl)))[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(str(c) for c in col if str(c) != "nan").strip()
                              for col in df.columns]
            else:
                df.columns = [str(c).strip() for c in df.columns]
            df.columns = make_unique_columns(df.columns.tolist())
            lines = []
            for _, row in df.head(3).iterrows():
                parts = [f"{c}:{str(row[c]).strip()}"
                         for c in df.columns
                         if str(row[c]).strip() not in ("nan", "-", "")]
                if parts:
                    lines.append(" / ".join(parts))
            text = " | ".join(lines)
    except Exception:
        pass
    cache.setdefault(horse_id, {})["train_text"] = text
    return text


# ── 出馬表から騎手・馬番・斤量等を取得 ─────────────────────────
def extract_jockey_id(href: str) -> str:
    """
    騎手URLから騎手IDを抽出。
    対応パターン:
      /jockey/05339/
      /jockey/result/recent/05339/
      /jockey/05339
    """
    if not href:
        return ""
    # 末尾の数字IDを取得（result/recent/ を含む場合も対応）
    m = re.search(r"/jockey/(?:[a-z/]*?)(\d{5})", href)
    if m:
        return m.group(1)
    # フォールバック：英数字ID
    m = re.search(r"/jockey/(?:result/recent/)?([A-Za-z0-9]+)/?$", href)
    return m.group(1) if m else ""


def parse_shutuba_table(table, soup: BeautifulSoup) -> dict:
    """
    出馬表テーブルから馬ごとの情報を返す。
    {馬名: {burden, age, jockey_name, jockey_id, gate_num}}
    """
    result = {}

    # pd.read_html で斤量・性齢・馬番を取得
    try:
        df_s = pd.read_html(io.StringIO(str(table)))[0]
        if isinstance(df_s.columns, pd.MultiIndex):
            df_s.columns = ["_".join(str(c) for c in col if str(c) != "nan").strip()
                            for col in df_s.columns]
        else:
            df_s.columns = [str(c).strip() for c in df_s.columns]
        df_s.columns = make_unique_columns(df_s.columns.tolist())

        burden_col = find_col(df_s, ["斤量"])
        age_col    = find_col(df_s, ["性齢"])
        name_col   = find_col(df_s, ["馬名"])
        gate_col   = find_col(df_s, ["枠番", "枠"])
        umaban_col = find_col(df_s, ["馬番"])

        if name_col:
            for _, row in df_s.iterrows():
                hn = str(row[name_col]).strip()
                if not hn or hn == "nan":
                    continue
                result[hn] = {
                    "burden":     float(pd.to_numeric(row.get(burden_col, 55), errors="coerce") or 55),
                    "age":        str(row.get(age_col, "")) if age_col else "",
                    "jockey_name": "未定",
                    "jockey_id":   "",
                    "gate_num":    0,
                }
                if umaban_col:
                    result[hn]["gate_num"] = int(pd.to_numeric(row[umaban_col], errors="coerce") or 0)
                elif gate_col:
                    result[hn]["gate_num"] = int(pd.to_numeric(row[gate_col], errors="coerce") or 0)
    except Exception:
        pass

    # HTMLから騎手名・騎手IDを取得（<a href="/jockey/"> を直接検索）
    try:
        for tr in table.find_all("tr"):
            horse_a = tr.find("a", href=re.compile(r"/horse/\d+"))
            if not horse_a:
                continue
            hn = horse_a.get_text(strip=True)
            if not hn:
                continue
            if hn not in result:
                result[hn] = {
                    "burden": 55.0, "age": "",
                    "jockey_name": "未定", "jockey_id": "", "gate_num": 0,
                }
            jockey_a = tr.find("a", href=re.compile(r"/jockey/"))
            if jockey_a:
                result[hn]["jockey_name"] = jockey_a.get_text(strip=True)
                result[hn]["jockey_id"]   = extract_jockey_id(jockey_a.get("href", ""))
            else:
                result[hn]["jockey_name"] = "未定"
    except Exception:
        pass

    return result


# ── レース情報（距離・馬場・競馬場）解析 ───────────────────────
def parse_race_info(soup: BeautifulSoup, url: str) -> tuple[str, str, int]:
    """(race_track, race_surface, race_distance) を返す"""
    race_distance, race_surface, race_track = 0, "芝", "汎用"

    for kw in ["RaceData01", "race_data", "mainrace_data"]:
        tag = soup.find(class_=kw)
        if tag:
            txt = tag.get_text()
            m = re.search(r"(\d{3,4})m", txt)
            if m:
                race_distance = int(m.group(1))
            if "ダ" in txt or "ダート" in txt:
                race_surface = "ダ"

    m_tr = re.search(r"race_id=\d{4}(\d{2})", url)
    if m_tr:
        race_track = PLACE_MAP.get(m_tr.group(1), "汎用")

    if race_distance == 0:
        m = re.search(r"(\d{3,4})m", soup.get_text())
        if m:
            race_distance = int(m.group(1))

    return race_track, race_surface, race_distance