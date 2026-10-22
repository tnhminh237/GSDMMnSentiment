import pandas as pd

v = pd.read_csv("/opt/data/raw/videos_nhac_ai.csv", encoding="utf-8-sig")
c = pd.read_parquet("/opt/data/gold/comments_clustered.parquet")

# Dấu hiệu AI trong tiêu đề: "AI" viết hoa, hoặc cụm viết thường rõ nghĩa
pat = r"\bA\.?I\b|suno|udio|trí tuệ nhân tạo|nhạc ai|ai cover|ai voice|ai music|ai hát|giọng ai"
v["title_ai"] = (v["title"].str.contains(pat, regex=True)
                 | v["title"].str.contains(r"(?i)suno|udio|trí tuệ nhân tạo|ai cover|ai voice"))

print("Video:", len(v), "| tiêu đề có dấu hiệu AI:", v["title_ai"].sum())
print("\nTỉ lệ tiêu đề có AI theo từ khóa tìm kiếm:")
print(v.groupby("keyword")["title_ai"].agg(["size", "mean"]))

# Gắn với bình luận
ai_c = c.assign(ai=c["text_token"].str.contains("tag_ai")).groupby("video_id")["ai"].mean()
v = v.merge(ai_c.rename("ai_comment_ratio"), left_on="video_id", right_index=True, how="left")
print("\nVideo KHÔNG có AI trong tiêu đề, 15 video ngẫu nhiên:")
print(v[~v["title_ai"]][["video_id", "title", "channel"]].sample(15, random_state=1).to_string())