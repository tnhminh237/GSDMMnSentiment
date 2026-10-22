import os
import numpy as np
import pandas as pd
from collections import Counter

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

K = 20
ALPHA = 0.1
BETA = 0.1
N_ITERS = 30
MIN_WORD_FREQ = 2
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
OUT = "" if SEED == 42 else f"_seed{SEED}"

RAW = "/opt/data/raw/other.parquet"
CAP_PER_VIDEO = None   
raw = pd.read_parquet(RAW)
funnel = [("raw", len(raw))]


df = raw[raw["is_modelable"]];           funnel.append(("is_modelable", len(df)))
df = df[~df["is_reply"]];                funnel.append(("bỏ reply", len(df)))
df = df.drop_duplicates("text_clean");   funnel.append(("bỏ trùng text_clean", len(df)))

# Lọc tiếng Việt: giữ bình luận có ít nhất 1 ký tự có dấu đặc trưng tiếng Việt
VI_CHARS = set("ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ")
def is_vi(s):
    return any(ch in VI_CHARS for ch in str(s).lower())
df = df[df["text_raw"].map(is_vi)];      funnel.append(("lọc tiếng Việt", len(df)))

# Giới hạn mỗi video (random_state cố định để mọi seed GSDMM dùng cùng một tập dữ liệu)
if CAP_PER_VIDEO:
    df = (df.groupby("video_id", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), CAP_PER_VIDEO), random_state=0)))
    funnel.append((f"cap {CAP_PER_VIDEO}/video", len(df)))
df = df.reset_index(drop=True)

# kiểm tra mức độ lệch theo video
vc = df["video_id"].value_counts()
print("Top 1 video chiếm:", round(vc.iloc[0] / len(df) * 100, 1), "%")
print("Top 10 video chiếm:", round(vc.head(10).sum() / len(df) * 100, 1), "%")
for name, n in funnel:
    print(f"{name:22s} {n}")
print("Số video còn lại:", df["video_id"].nunique())

def to_text(t):
    if isinstance(t, str):
        return t
    return " ".join(map(str, t))
df["text"] = df["text_raw"]
df["text_gsdmm"] = df["tokens"].map(to_text)

print("Mẫu tokens:")
for t in df["text_gsdmm"].sample(5, random_state=1):
    print("  ", t[:100])

# 1. Xây từ điển
docs = [t.split() for t in df["text_gsdmm"]]
freq = Counter(w for d in docs for w in d)
vocab = {w: i for i, w in enumerate(w for w, c in freq.items() if c >= MIN_WORD_FREQ)}
inv_vocab = {i: w for w, i in vocab.items()}
V = len(vocab)

# 2. Mỗi bình luận -> (id từ, số lần xuất hiện)
docs_ids, keep = [], []
for d in docs:
    c = Counter(w for w in d if w in vocab)
    if sum(c.values()) >= 2:
        ids = np.array([vocab[w] for w in c], dtype=int)
        cs = np.array(list(c.values()), dtype=int)
        docs_ids.append((ids, cs))
        keep.append(True)
    else:
        keep.append(False)
df = df[keep].reset_index(drop=True)
D = len(docs_ids)
print(f"Số bình luận: {D}, kích thước từ điển: {V}")

# 3. Khởi tạo ngẫu nhiên
rng = np.random.default_rng(SEED)
z = rng.integers(0, K, size=D)
m = np.zeros(K)            # số bình luận mỗi cụm
n = np.zeros(K)            # tổng số từ mỗi cụm
nkw = np.zeros((K, V))     # số lần từ w xuất hiện trong cụm k
for d, (ids, cs) in enumerate(docs_ids):
    k = z[d]
    m[k] += 1; n[k] += cs.sum(); nkw[k, ids] += cs

def log_prob(ids, cs):
    lp = np.log(m + ALPHA)
    for w, c in zip(ids, cs):
        for j in range(c):
            lp += np.log(nkw[:, w] + BETA + j)
    for i in range(cs.sum()):
        lp -= np.log(n + V * BETA + i)
    return lp

# 4. Gibbs sampling
for it in range(N_ITERS):
    moved = 0
    for d in rng.permutation(D):
        ids, cs = docs_ids[d]
        k = z[d]
        m[k] -= 1; n[k] -= cs.sum(); nkw[k, ids] -= cs
        lp = log_prob(ids, cs)
        p = np.exp(lp - lp.max()); p /= p.sum()
        k2 = min(int(np.searchsorted(np.cumsum(p), rng.random())), K - 1)
        if k2 != k:
            moved += 1
        z[d] = k2
        m[k2] += 1; n[k2] += cs.sum(); nkw[k2, ids] += cs
    print(f"Vòng {it+1}/{N_ITERS}: {moved} bình luận đổi cụm, còn {(m > 0).sum()} cụm")

# 5. Lưu kết quả
os.makedirs("/opt/data/gold", exist_ok=True)
df["cluster"] = z
df.to_parquet(f"/opt/data/gold/comments_clustered{OUT}.parquet", index=False)
np.savez(f"/opt/data/gold/gsdmm_model{OUT}.npz", m=m, n=n, nkw=nkw,
         vocab=np.array(list(vocab.keys())))

rows = []
for k in np.argsort(-m):
    if m[k] == 0:
        continue
    col_tot = nkw.sum(axis=0) + 1
    score = (nkw[k] / col_tot) * np.log1p(nkw[k])   # từ tập trung ở cụm này, và đủ phổ biến
    top = [inv_vocab[i] for i in np.argsort(-score)[:15]]
    rows.append({"cluster": int(k), "so_binh_luan": int(m[k]), "top_words": " ".join(top)})
    print(f"\n=== Cụm {k} ({int(m[k])} bình luận) ===")
    print("Từ khóa:", " ".join(top))
    for t in df.loc[df["cluster"] == k, "text"].head(3):
        print("  -", str(t).replace("\n", " ")[:100])
pd.DataFrame(rows).to_csv("/opt/data/gold/cluster_top_words.csv", index=False, encoding="utf-8-sig")