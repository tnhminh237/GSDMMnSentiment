import re
import pandas as pd

v = pd.read_csv("/opt/data/raw/videos_nhac_ai_enriched.csv", encoding="utf-8-sig").fillna("")
UPPER = re.compile(r"\bA\.?I\b")
PHRASE = re.compile(r"suno|\budio\b|trí tuệ nhân tạo|ai cover|ai voice|ai music|"
                    r"ai-generated|ai generated|nhạc ai|giọng ai", re.I)

def has_ai(s):
    return bool(UPPER.search(s) or PHRASE.search(s))

def context(s):
    m = UPPER.search(s) or PHRASE.search(s)
    if not m:
        return ""
    a, b = max(0, m.start() - 50), min(len(s), m.end() + 50)
    return s[a:b].replace("\n", " ")

v["strong"] = v["title"].map(has_ai) | v["channel"].map(has_ai)
v["weak"] = ~v["strong"] & (v["description"].map(has_ai) | v["tags"].map(has_ai))
print("Mạnh (tiêu đề/kênh):", v["strong"].sum(), "| Yếu (chỉ mô tả/tag):", v["weak"].sum(),
      "| Không:", (~v["strong"] & ~v["weak"]).sum())

print("\n=== Ngữ cảnh chữ AI trong mô tả của 20 video 'yếu' ===")
for _, r in v[v["weak"]].sample(20, random_state=2).iterrows():
    print(f"- {r['title'][:50]}")
    print(f"    ...{context(r['description']) or ('[tag] ' + r['tags'][:100])}...")

v[["video_id", "strong", "weak"]].to_csv("/opt/data/gold/video_ai_filter.csv", index=False)