import pandas as pd

df = pd.read_parquet("/opt/data/gold/comments_clustered.parquet")
df["ai"] = df["text_token"].str.contains("tag_ai")

g = df.groupby("video_id").agg(so_bl=("ai", "size"), n_ai=("ai", "sum"))
g["ai_ratio"] = g["n_ai"] / g["so_bl"]
print("Video có nhắc AI:", (g["n_ai"] > 0).sum(), "/", len(g))
print(g.sort_values("n_ai", ascending=False).head(15))
print("\nTỉ lệ nhắc AI theo cụm:")
print(df.groupby("cluster")["ai"].mean().sort_values(ascending=False).head(8))