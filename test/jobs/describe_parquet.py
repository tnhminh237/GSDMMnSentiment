import pandas as pd

df = pd.read_parquet("/opt/data/raw/other.parquet")
pd.set_option("display.width", 200, "display.max_colwidth", 70)
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print(df["source"].value_counts())
print(df["is_reply"].value_counts())
print(df["is_modelable"].value_counts())
print("Số video:", df["video_id"].nunique())
print(df[["video_id", "text_raw", "text_clean", "n_tokens", "is_modelable"]].sample(12, random_state=1))