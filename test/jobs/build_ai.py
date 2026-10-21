import re
import pandas as pd

v = pd.read_csv("/opt/data/raw/videos_nhac_ai_enriched.csv", encoding="utf-8-sig").fillna("")

UPPER = re.compile(r"\bA\.?I\b")
CLEAR = re.compile(r"suno|\budio\b|trí tuệ nhân tạo|ai cover|ai voice|ai music|"
                   r"ai-generated|ai generated|nhạc ai|giọng ai|"
                   r"(công nghệ|bằng|bởi|minh bạch|tạo bởi|nhờ) ai\b|#aimusic|sunoai", re.I)

def mostly_upper(s):
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.7

def title_ai(s):
    if CLEAR.search(s):
        return True
    return bool(UPPER.search(s)) and not mostly_upper(s)

v["sig_title"] = v["title"].map(title_ai)
v["sig_channel"] = v["channel"].map(lambda s: bool(CLEAR.search(s) or UPPER.search(s)))
v["sig_desc"] = v["description"].map(lambda s: bool(CLEAR.search(s)))

v["ai_final"] = v["sig_title"] | v["sig_channel"] | v["sig_desc"]

print("Tổng video:", len(v))
print("Giữ lại (ai_final):", v["ai_final"].sum())
print(v[["sig_title", "sig_channel", "sig_desc"]].sum())
print("\nTheo từ khóa tìm kiếm:")
print(v.groupby("keyword")["ai_final"].agg(["size", "mean"]))

print("\n10 video GIỮ vì tiêu đề viết hoa + có AI (kiểm tra dương tính giả):")
chk = v[v["ai_final"] & v["title"].map(mostly_upper)]
print(chk[["title", "channel"]].head(10).to_string())

v[["video_id", "title", "channel", "keyword", "views", "likes", "ai_final"]] \
    .to_csv("/opt/data/gold/video_ai_final.csv", index=False, encoding="utf-8-sig")