import pandas as pd

D = "/opt/data"
raw   = pd.read_csv(f"{D}/raw/comments.csv")
clean = pd.read_parquet(f"{D}/silver/comments_clean")
tok   = pd.read_parquet(f"{D}/silver/comments_tokenized")
vid   = pd.read_csv(f"{D}/gold/video_ai_final.csv")

print("1. raw:", len(raw))
print("   comment_id trùng trong raw:", raw["comment_id"].duplicated().sum())
print("   số video có bình luận (raw):", raw["video_id"].nunique())

print("2. clean:", len(clean))
print("   số video có bình luận (clean):", clean["video_id"].nunique())

ai_ids = set(vid.loc[vid["ai_final"] == 1, "video_id"])
print("3. video AI (ai_final=1):", len(ai_ids))
print("   video AI có bình luận trong raw  :", raw["video_id"].isin(ai_ids).sum(), "dòng")
print("   video AI có bình luận trong clean:", clean["video_id"].isin(ai_ids).sum(), "dòng")

print("4. tokenized:", len(tok))
print("   số video (tokenized):", tok["video_id"].nunique())

# Phân bố số bình luận mỗi video AI
cnt = clean[clean["video_id"].isin(ai_ids)].groupby("video_id").size()
print("\nBình luận/video AI (sau clean):")
print(cnt.describe())