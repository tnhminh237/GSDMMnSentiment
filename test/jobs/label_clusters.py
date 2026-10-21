import pandas as pd

df = pd.read_parquet("/opt/data/gold/comments_clustered.parquet")
GROUP = {
    9: "Khen & cảm ơn", 2: "Khen & cảm ơn",
    4: "Xin karaoke/beat",
    3: "Phản ứng với AI & cách dùng",
    6: "Tranh luận AI-nghệ sĩ, bản quyền, thương mại",
    0: "Hoài niệm & văn thơ", 8: "Hoài niệm & văn thơ",
}
df["nhom"] = df["cluster"].map(GROUP)      # NaN = cụm nhiễu, loại
print("Loại:", df["nhom"].isna().sum(), "bình luận")
df = df.dropna(subset=["nhom"]).copy()
df["nhac_ai"] = df["text_token"].str.contains("tag_ai")

print(df.groupby("nhom").agg(so_bl=("nhom", "size"), ai_pct=("nhac_ai", "mean")).round(3))
df.to_parquet("/opt/data/gold/comments_labeled.parquet", index=False)