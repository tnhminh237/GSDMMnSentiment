import pandas as pd

df = pd.read_parquet("/opt/data/gold/comments_clustered.parquet")
df["ai"] = df["text_token"].str.contains("tag_ai")
g = df.groupby("cluster").agg(so_bl=("ai", "size"), ai_pct=("ai", "mean"))
g["ai_pct"] = (g["ai_pct"] * 100).round(1)
g = g.sort_values("so_bl", ascending=False)
print(g)

for k in g.index:
    sub = df[df["cluster"] == k]
    print(f"\n=== Cụm {k} ({len(sub)} BL) ===")
    for x in sub["text"].sample(min(8, len(sub)), random_state=7):
        print("  -", str(x).replace("\n", " ")[:110])