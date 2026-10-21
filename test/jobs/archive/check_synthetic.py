import pandas as pd

v = pd.read_csv("/opt/data/raw/videos_nhac_ai_enriched.csv", encoding="utf-8-sig")
print(v["synthetic"].value_counts(dropna=False))
print()
print(v[["title", "channel", "synthetic"]].head(10).to_string())