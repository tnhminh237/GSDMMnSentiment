import pandas as pd

RAW = "/opt/data/raw/other.parquet"
CAP_PER_VIDEO = None   # None = không giới hạn; đặt 100 nếu muốn giảm lệch theo video

# ---------- 1. Đọc và lọc ----------
df = pd.read_parquet(RAW)
funnel = [("raw", len(df))]
df = df[df["is_modelable"]];            funnel.append(("is_modelable", len(df)))
df = df[~df["is_reply"]];               funnel.append(("bỏ reply", len(df)))
df = df.drop_duplicates("text_raw");    funnel.append(("bỏ trùng", len(df)))

# Lọc tiếng Việt: giữ bình luận có ít nhất 1 ký tự có dấu đặc trưng
VI_CHARS = set("ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ")
df = df[df["text_raw"].map(lambda s: any(c in VI_CHARS for c in str(s).lower()))]
funnel.append(("lọc tiếng Việt", len(df)))

# Giới hạn tối đa bình luận mỗi video (tùy chọn, cố định random_state để tái lập)
if CAP_PER_VIDEO:
    df = df.sample(frac=1, random_state=0).groupby("video_id").head(CAP_PER_VIDEO)
    funnel.append((f"cap {CAP_PER_VIDEO}/video", len(df)))

df = df.reset_index(drop=True)
for k, v in funnel:
    print(f"{k:16s} {v}")

vc = df["video_id"].value_counts()
print(f"Số video: {df['video_id'].nunique()}")
print(f"Top 1 video chiếm: {vc.iloc[0] / len(df) * 100:.1f}% | "
      f"Top 10 video chiếm: {vc.head(10).sum() / len(df) * 100:.1f}%")

df["text"] = df["text_raw"].astype(str)
t = df["text"].str.lower()

# ---------- 2. Từ điển khía cạnh ----------
# Giả định: mọi bình luận trong other.parquet thuộc video nhạc AI.
ASPECTS = {
    "giong_hat": r"giọng|giống thật|giống y|giống hệt|giống bản gốc|y chang|y hệt|như thật|nghe giống|giống quá",
    "am_thanh": r"âm thanh|chất lượng|\bmix\b|\bmaster\b|\bbass\b|nhạc nền|hòa âm|phối khí|bản phối|âm lượng|tạp âm|\brè\b",
    "sang_tao": r"sáng tạo|ý tưởng|độc đáo|mới lạ|phối mới|làm mới|cách làm|phong cách",
    "ban_quyen": r"bản quyền|copyright|đạo nhạc|xin phép|ăn cắp|mạo danh|giả mạo|deepfake|giả giọng",
    "dao_duc": r"mất việc|thất nghiệp|cướp việc|đạo đức|tôn trọng|nghệ sĩ thật|\brác\b|vô hồn|lười|thay thế",
}
OTHERS = {
    "khen_bai_goc": r"sáng tác|lời bài|giai điệu|ca từ|tác giả|nhạc sĩ|lời ca",
    "yeu_cau_karaoke": r"karaoke|kara|\bbeat\b|instrumental|xin file|link tải|tải về|download|xin bản",
    "yeu_cau_bai_moi": r"làm bài|làm thêm|làm tiếp|ra bài|ra thêm|cover bài|làm lại|làm giúp|tag em|tag mình",
    "hoi_dap_ky_thuat": r"hướng dẫn|làm sao|cách tạo|prompt|upload|tài khoản|account|credit|mua gói|trả phí",
    "khen_chung": r"tuyệt vời|tuyệt|\bhay\b|đỉnh|cảm ơn|hay quá|thích|xuất sắc|quá đã|cảm xúc|ý nghĩa",
}
ALL = {**ASPECTS, **OTHERS}

# ---------- 3. Gán khía cạnh (một bình luận có thể thuộc nhiều khía cạnh) ----------
n = len(df)
print(f"\nTổng bình luận: {n}\n")

for name, pat in ALL.items():
    df[name] = t.str.contains(pat, regex=True)
df["khong_khia_canh"] = ~df[list(ALL)].any(axis=1)

# ---------- 4. Thống kê ----------
print(f"{'khia_canh':18s} {'so_cmt':>8s} {'%':>6s}")
for name in list(ALL) + ["khong_khia_canh"]:
    a = int(df[name].sum())
    print(f"{name:18s} {a:8d} {a/n*100:5.1f}%")

print("\n--- 5 bình luận mẫu mỗi khía cạnh ---")
for name in list(ALL) + ["khong_khia_canh"]:
    sub = df[df[name]]["text"]
    print(f"\n[{name}]")
    for x in sub.sample(min(5, len(sub)), random_state=1):
        print("  -", str(x).replace("\n", " ")[:110])

# ---------- 5. Ghi kết quả ----------
df.to_parquet("/opt/data/gold/comments_aspects.parquet", index=False)
print("\nĐã ghi gold/comments_aspects.parquet")