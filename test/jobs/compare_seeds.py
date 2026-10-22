import pandas as pd

a = pd.read_parquet("/opt/data/gold/comments_clustered.parquet")[["comment_id", "cluster"]]
a = a.rename(columns={"cluster": "c42"})
for s in (1, 7):
    b = pd.read_parquet(f"/opt/data/gold/comments_clustered_seed{s}.parquet")[["comment_id", "cluster"]]
    a = a.merge(b.rename(columns={"cluster": f"c{s}"}), on="comment_id")

for s in (1, 7):
    ct = pd.crosstab(a["c42"], a[f"c{s}"])
    share = ct.div(ct.sum(axis=1), axis=0)
    out = pd.DataFrame({"so_bl": ct.sum(axis=1),
                        f"cum_seed{s}_chinh": share.idxmax(axis=1),
                        "ty_le_cung_cum": share.max(axis=1).round(2)})
    print(f"\nSeed 42 so với seed {s}:")
    print(out)