import re
import pandas as pd

v = pd.read_csv("/opt/data/raw/videos_nhac_ai_enriched.csv", encoding="utf-8-sig").fillna("")
print("Cột:", list(v.columns))
print("Mô tả không rỗng:", (v["description"] != "").sum(), "| tag không rỗng:", (v["tags"] != "").sum())

# "AI" viết hoa (trí tuệ nhân tạo), khác "Ai" = who
UPPER = re.compile(r"\bA\.?I\b")
# Cụm viết thường rõ nghĩa (\budio\b để không dính "audio")
PHRASE = re.compile(r"suno|\budio\b|trí tuệ nhân tạo|ai cover|ai voice|ai music|"
                    r"ai-generated|ai generated|nhạc ai|giọng ai", re.I)

def has_ai(s):
    return bool(UPPER.search(s) or PHRASE.search(s))

for col in ["title", "channel", "description", "tags"]:
    v["ai_" + col] = v[col].map(has_ai)

v["ai_any"] = v[["ai_title", "ai_channel", "ai_description", "ai_tags"]].any(axis=1)

print("\nSố video có tín hiệu AI theo từng trường:")
print(v[["ai_title", "ai_channel", "ai_description", "ai_tags", "ai_any"]].sum())
print("\nTỉ lệ video có tín hiệu AI theo từ khóa tìm kiếm:")
print(v.groupby("keyword")["ai_any"].agg(["size", "mean"]))

# Mẫu để tự kiểm tra bằng mắt
only_desc = v[v["ai_any"] & ~v["ai_title"] & ~v["ai_channel"]]
print("\nChỉ có AI trong mô tả/tag (10 mẫu):")
print(only_desc[["title", "channel"]].sample(min(10, len(only_desc)), random_state=1).to_string())
print("\nKhông có tín hiệu AI nào (10 mẫu):")
print(v[~v["ai_any"]][["title", "channel"]].sample(10, random_state=1).to_string())

v[["video_id", "ai_any"]].to_csv("/opt/data/gold/video_ai_filter.csv", index=False)