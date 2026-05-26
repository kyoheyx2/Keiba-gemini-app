"""
ai.py — Gemini API 呼び出し・プロンプト生成・レスポンス解析
"""
import json, re, time
import pandas as pd
import requests
import streamlit as st

from config import GEMINI_MODEL

TICKET_ICONS = {
    "単勝":"🥇","複勝":"🏅","馬連":"🔗","馬単":"⚡",
    "ワイド":"🎯","三連複":"🎰","三連単":"💥",
}


def get_gemini_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""


def call_gemini(prompt: str, api_key: str) -> str:
    if not api_key:
        return "⚠️ GEMINI_API_KEY が未設定です（サイドバーで入力してください）"
    for attempt in range(6):
        try:
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{GEMINI_MODEL}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8192},
                },
                timeout=90,
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in (503, 429):
                time.sleep((2 ** attempt) * 5)
                continue
            return f"⚠️ APIエラー ({e.response.status_code})"
        except Exception:
            if attempt < 5:
                time.sleep(3)
                continue
    return "❌ Gemini APIが混雑しています。しばらく待ってから再試行してください。"


def build_analysis_prompt(race_name: str, result_df: pd.DataFrame) -> str:
    n = len(result_df)
    rows = []
    for _, r in result_df.iterrows():
        train = str(r.get("調教内容", "")).strip()
        train_str = f" 【調教】{train[:50]}" if train else ""
        rows.append(
            f"  {r['馬名']}: 総合{r['総合指数']} "
            f"速度{r['スピード指数']} 能力{r['能力指数']} "
            f"EB{r['EB指数']} Base{r['Base指数']} "
            f"騎手{r.get('騎手指数','-')} 間隔{r.get('間隔スコア','-')} "
            f"上がり{r.get('上がり3F','-')}{train_str}"
        )
    horse_lines = "\n".join(rows)
    template = "\n".join(
        f'    {{"馬名": "{r["馬名"]}", "勝率": 0.0, "連対率": 0.0, "複勝率": 0.0, "コメント": ""}}'
        + ("," if i < n - 1 else "")
        for i, (_, r) in enumerate(result_df.iterrows())
    )
    return f"""競馬専門アナリストとして「{race_name}」（{n}頭立て）を分析し、
馬券の買い方まで提案してください。

【7指数の説明】
①スピード: 西田式（タイム×距離指数+馬場補正+斤量補正+80）
②能力: スピード指数×クラス補正×着順補正×枠番補正
③EB: Beta-Binomial shrinkage（連対率の安定推定）
④Base: コース・距離適性
⑤騎手: 当該コース・馬場の勝率・連対率から指数化
⑥間隔: 前走間隔（叩き2走目=高評価）
⑦上がり3F: 末脚の質（33.5秒基準）

【出走馬データ（全{n}頭・レース内正規化済み）】
{horse_lines}

【確率設定指針】
- 総合指数上位馬を中心に勝率設定（1位は最低15%以上）
- 全馬の勝率合計=100%
- 連対率≈勝率×2倍、複勝率≈勝率×3倍

【馬券推奨方針】
- 本命（◎）: 総合指数が最も高い馬
- 対抗（○）: 本命に次ぐ有力馬
- 穴馬（△）: 指数より人気が低く期待値が高い馬
- 推奨馬券: 単勝/複勝/馬連/ワイド/三連複から最も回収率が高い買い方

【出力ルール】JSONのみ（コードブロック禁止）・全{n}頭必須・勝率合計=100

{{
  "horses": [
{template}
  ],
  "レース総評": "(50字以内)",
  "本命": "(馬名)",
  "対抗": "(馬名)",
  "穴馬": ["(馬名1)", "(馬名2)"],
  "推奨馬券": [
    {{"券種": "馬連", "組み合わせ": "(本命)-(対抗)", "理由": "(30字以内)", "金額比率": "50%"}},
    {{"券種": "ワイド", "組み合わせ": "(本命)-(穴馬1)", "理由": "(30字以内)", "金額比率": "30%"}},
    {{"券種": "単勝", "組み合わせ": "(本命)", "理由": "(30字以内)", "金額比率": "20%"}}
  ]
}}"""


def parse_ai_response(text: str):
    text = re.sub(r"```+\w*\s*", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        cand = m.group(1)
        try:
            return json.loads(cand)
        except Exception:
            for tail in ["]}}","]}","}"]:
                try:
                    return json.loads(cand + tail)
                except Exception:
                    pass
    return None


def _to_pct(val) -> float:
    """
    AIが返す確率値を % スケール（0〜100）に正規化する。
    - 0.25 のような小数 → 25.0 に変換
    - 25.0 のような % 表記 → そのまま
    - "25%" のような文字列 → 25.0 に変換
    """
    try:
        v = float(str(val).replace("%", "").strip())
        # 1.0 未満は小数表記と判断して ×100
        if v < 1.0:
            v *= 100.0
        return round(v, 1)
    except Exception:
        return 0.0


def render_ai_result(ai_data: dict, ai_text: str):
    """AI分析結果をStreamlitで表示"""
    if not (ai_data and "horses" in ai_data):
        st.subheader("🤖 AI分析（Gemini 2.5 Flash）")
        st.markdown(ai_text)
        return

    st.subheader("🤖 AI分析（Gemini 2.5 Flash）")
    if "レース総評" in ai_data:
        st.info(f"📝 {ai_data['レース総評']}")

    ai_df_data = [{
        "馬名":      h.get("馬名",""),
        "勝率(%)":   _to_pct(h.get("勝率",  0)),
        "連対率(%)": _to_pct(h.get("連対率", 0)),
        "複勝率(%)": _to_pct(h.get("複勝率", 0)),
        "AIコメント": h.get("コメント",""),
    } for h in ai_data["horses"]]
    ai_df = pd.DataFrame(ai_df_data).sort_values("勝率(%)", ascending=False)
    ai_df.index = range(1, len(ai_df)+1)
    ai_df.index.name = "順位"

    st.markdown("**📈 勝率・連対率・複勝率（AI推定）**")
    st.dataframe(
        ai_df[["馬名","勝率(%)","連対率(%)","複勝率(%)"]],
        use_container_width=True,
        height=min(55 + len(ai_df)*36, 620),
    )
    with st.expander("💬 AIコメント一覧"):
        for rank, row in ai_df.iterrows():
            st.markdown(f"**{rank}位 {row['馬名']}**：{row['AIコメント']}")

    st.divider()
    st.subheader("🎯 AI推奨買い目")
    honmei = ai_data.get("本命","")
    taikou = ai_data.get("対抗","")
    ana    = ai_data.get("穴馬",[])
    if isinstance(ana, str):
        ana = [ana]
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("◎ 本命", honmei)
    with c2: st.metric("○ 対抗", taikou)
    with c3: st.metric("△ 穴馬", " / ".join(ana) if ana else "−")

    for b in ai_data.get("推奨馬券", []):
        icon = TICKET_ICONS.get(b.get("券種",""), "🎫")
        st.success(
            f"{icon} **{b.get('券種','')}**　{b.get('組み合わせ','')}　"
            f"予算比率: {b.get('金額比率','')}　{b.get('理由','')}"
        )