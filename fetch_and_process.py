# -*- coding: utf-8 -*-
"""
个人信息源 Agent · 抓取管线 (M1)
流程: 读 sources.txt -> 抓 RSS -> 规则初筛 -> 大模型打分+压缩(单条一次调用)
      -> 写 data/feed.json / data/history.json -> 3星条目推送飞书

环境变量(在 GitHub Secrets 或本地 .env 配置):
  OPENAI_API_KEY   必填, 大模型 Key
  OPENAI_BASE_URL  选填, 默认 https://api.openai.com/v1 (兼容所有OpenAI格式接口)
  MODEL_NAME       选填, 默认 gpt-4o-mini
  USER_DOMAINS     选填, 关注领域, 默认 "美股交易,宏观经济,AI,科技,跨境电商"
  SCORE_THRESHOLD  选填, 默认 6
  FEISHU_WEBHOOK   选填, 飞书自定义机器人 webhook 地址, 不填则不推送
  MOCK_LLM         选填, 设为 1 时不调大模型(本地测试用, 用规则近似打分)
"""
import os, re, json, time, hashlib, html
from datetime import datetime, timezone, timedelta

import requests
import feedparser

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- 配置 ----------------
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
MODEL = os.environ.get("MODEL_NAME") or "gpt-4o-mini"
DOMAINS = os.environ.get("USER_DOMAINS") or "美股交易,宏观经济,AI,科技,跨境电商"
THRESHOLD = float(os.environ.get("SCORE_THRESHOLD") or "6")
FEISHU = os.environ.get("FEISHU_WEBHOOK", "").strip()
MOCK = os.environ.get("MOCK_LLM", "") == "1"

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)
FEED_PATH = os.path.join(DATA, "feed.json")
HIST_PATH = os.path.join(DATA, "history.json")

JUNK_WORDS = ["震惊", "必看", "炸裂", "速看", "惊呆", "小编", "家人们", "宝子们", "点赞关注"]
CST = timezone(timedelta(hours=8))

MODULE_MAP = {"美股": "美股投资", "美股投资": "美股投资", "新闻": "新闻监控", "新闻监控": "新闻监控"}

# ---------------- 工具 ----------------
def now_iso():
    return datetime.now(CST).strftime("%Y-%m-%dT%H:%M:%S+08:00")

def strip_html(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    return html.unescape(re.sub(r"\s+", " ", s)).strip()

def item_id(link, title):
    return hashlib.md5((link or title).encode("utf-8")).hexdigest()[:16]

def tokens(text):
    # 中英混排: 中文按二字滑窗, 英文按单词
    zh = re.findall(r"[\u4e00-\u9fff]", text)
    grams = {"".join(zh[i:i+2]) for i in range(len(zh) - 1)}
    en = set(re.findall(r"[a-zA-Z]{3,}", text.lower()))
    return grams | en

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)

# ---------------- 1. 读信源 ----------------
def read_sources():
    path = os.path.join(ROOT, "sources.txt")
    module = "新闻监控"
    out = []
    if not os.path.exists(path):
        print("[warn] sources.txt 不存在")
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                tag = line.lstrip("#").strip()
                module = MODULE_MAP.get(tag, tag or module)
                continue
            out.append((module, line))
    return out

# ---------------- 2. 抓取 ----------------
def fetch_all(sources, seen_ids):
    items = []
    for module, url in sources:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 InfoAgent/1.0"})
            fp = feedparser.parse(r.content)
            name = strip_html(getattr(fp.feed, "title", "")) or url
            for e in fp.entries[:20]:
                link = e.get("link", "")
                title = strip_html(e.get("title", ""))
                iid = item_id(link, title)
                if not title or iid in seen_ids:
                    continue
                body = ""
                if e.get("content"):
                    body = strip_html(e["content"][0].get("value", ""))
                body = body or strip_html(e.get("summary", ""))
                items.append({
                    "id": iid, "module": module, "source": name,
                    "title": title, "content": body[:4000], "url": link,
                })
        except Exception as ex:
            print(f"[warn] 抓取失败 {url}: {ex}")
    return items

# ---------------- 3. 规则初筛 ----------------
def prefilter(items, history):
    recent = [h for h in history
              if h.get("ts", "") >= (datetime.now(CST) - timedelta(days=7)).strftime("%Y-%m-%dT")]
    recent_tok = [tokens(h.get("title", "") + h.get("core_conclusion", "")) for h in recent]
    kept = []
    for it in items:
        text = it["title"] + it["content"]
        if len(text) < 100:
            continue
        if any(w in text for w in JUNK_WORDS):
            continue
        tk = tokens(it["title"])
        if any(jaccard(tk, rt) >= 0.6 for rt in recent_tok if rt):
            continue
        kept.append(it)
    return kept

# ---------------- 4. 大模型: 打分+压缩 一次完成 ----------------
PROMPT = """你是严格的信息价值打分与压缩器。只返回JSON数组,无其他内容,无markdown围栏。
用户关注领域: {domains}
对下面每条信息输出一个对象:
{{"id":"原样返回","score":0到10,"is_junk":true或false,
"core_conclusion":"一句话核心结论,不超过30字",
"key_data":["硬数据或事实,最多3条,无则空数组"],
"logic_chain":["根源逻辑推理,最多2步"],
"anchor_q":"一个针对该信息、能触发用户独立判断的思考问题,不超过30字"}}
打分: 相关性0-4 + 信息密度0-2.5 + 新颖性0-2 + 可行动性0-1.5。
广告/软文/纯情绪/无事实内容 is_junk=true。
压缩要求: 删除所有形容词、情绪词、铺垫; key_data只留可验证硬事实。

信息列表:
{items}"""

def call_llm(batch):
    if MOCK:
        return [{
            "id": it["id"], "score": min(10.0, 6.0 + len(it["content"]) / 80.0),
            "is_junk": False,
            "core_conclusion": it["title"][:30],
            "key_data": [it["content"][:60]] if it["content"] else [],
            "logic_chain": ["MOCK模式本地测试数据"],
            "anchor_q": "这条信息会改变你的哪个判断?",
        } for it in batch]
    payload = {
        "model": MODEL,
        "max_tokens": 2000,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": PROMPT.format(
            domains=DOMAINS,
            items=json.dumps(
                [{"id": b["id"], "标题": b["title"], "内容": b["content"][:1500]} for b in batch],
                ensure_ascii=False),
        )}],
    }
    r = requests.post(f"{BASE_URL}/chat/completions", timeout=90,
                      headers={"Authorization": f"Bearer {API_KEY}",
                               "Content-Type": "application/json"},
                      json=payload)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)

def score_and_compress(items):
    out = []
    for i in range(0, len(items), 4):          # 每次调用带4条, 控制调用量
        batch = items[i:i+4]
        try:
            results = {r["id"]: r for r in call_llm(batch)}
        except Exception as ex:
            print(f"[warn] LLM 调用失败, 跳过该批: {ex}")
            continue
        for it in batch:
            r = results.get(it["id"])
            if not r or r.get("is_junk") or float(r.get("score", 0)) < THRESHOLD:
                continue
            score = float(r["score"])
            star = 3 if score >= 8 else (2 if score >= 7 else 1)
            out.append({
                "id": it["id"], "module": it["module"], "source": it["source"],
                "title": it["title"], "url": it["url"],
                "score": round(score, 1), "star": star,
                "core_conclusion": r.get("core_conclusion", "")[:60],
                "key_data": (r.get("key_data") or [])[:3],
                "logic_chain": (r.get("logic_chain") or [])[:2],
                "anchor_q": r.get("anchor_q", ""),
                "ts": now_iso(),
                # M2 预留
                "source_type": "rss", "push_date": now_iso()[:10], "feedback": "",
            })
        time.sleep(1)
    return out

# ---------------- 5. 飞书推送(3星) ----------------
def push_feishu(items):
    stars3 = [x for x in items if x["star"] == 3]
    if not FEISHU or not stars3:
        return 0
    n = 0
    for it in stars3[:5]:
        lines = [f"★★★ [{it['module']}] {it['core_conclusion']}"]
        if it["key_data"]:
            lines.append("关键数据: " + " | ".join(it["key_data"]))
        if it["logic_chain"]:
            lines.append("根源逻辑: " + " → ".join(it["logic_chain"]))
        lines.append("原文: " + it["url"])
        try:
            requests.post(FEISHU, timeout=15,
                          json={"msg_type": "text", "content": {"text": "\n".join(lines)}})
            n += 1
        except Exception as ex:
            print(f"[warn] 飞书推送失败: {ex}")
    return n

# ---------------- 主流程 ----------------
def main():
    history = load_json(HIST_PATH, [])
    seen = {h["id"] for h in history}
    sources = read_sources()
    print(f"信源 {len(sources)} 个")
    raw = fetch_all(sources, seen)
    print(f"新条目 {len(raw)}")
    kept = prefilter(raw, history)
    print(f"初筛通过 {len(kept)}")
    kept = kept[:24]                            # 单轮上限, 控成本
    final = score_and_compress(kept)
    print(f"打分通过 {len(final)}")

    feed = load_json(FEED_PATH, {"items": []})
    items = final + feed.get("items", [])
    cutoff = (datetime.now(CST) - timedelta(days=3)).strftime("%Y-%m-%dT")
    items = [x for x in items if x["ts"] >= cutoff][:200]   # 页面保留3天
    save_json(FEED_PATH, {
        "updated": now_iso(), "total": len(items),
        "high_priority": sum(1 for x in items if x["star"] == 3),
        "items": items,
    })

    # 历史: 全量id + 摘要, 保留30天
    hcut = (datetime.now(CST) - timedelta(days=30)).strftime("%Y-%m-%dT")
    for it in raw:
        history.append({"id": it["id"], "title": it["title"],
                        "core_conclusion": "", "ts": now_iso()})
    for x in final:
        for h in history:
            if h["id"] == x["id"]:
                h["core_conclusion"] = x["core_conclusion"]
    history = [h for h in history if h.get("ts", "") >= hcut][-3000:]
    save_json(HIST_PATH, history)

    pushed = push_feishu(final)
    print(f"飞书推送 {pushed} 条, 完成")

if __name__ == "__main__":
    main()
