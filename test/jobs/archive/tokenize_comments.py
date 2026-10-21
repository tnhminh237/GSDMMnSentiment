import re
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import pandas_udf

spark = SparkSession.builder.appName("tokenize-comments").getOrCreate()

df = spark.read.parquet("/opt/data/silver/comments_clean")
ai_vids = (spark.read.option("header", True).csv("/opt/data/gold/video_ai_final.csv")
           .filter(F.col("ai_final") == "True").select("video_id"))
df = df.join(ai_vids, "video_id")
print("Số video AI có bình luận:", df.select("video_id").distinct().count())

# 1. Emoji -> nhãn cảm xúc
EMOJI_MAP = {
    "😂": " emo_cuoi ", "🤣": " emo_cuoi ", "😆": " emo_cuoi ", "😝": " emo_cuoi ",
    "😍": " emo_yeu ", "🥰": " emo_yeu ", "❤": " emo_yeu ", "♥": " emo_yeu ",
    "💖": " emo_yeu ", "💙": " emo_yeu ", "🩵": " emo_yeu ", "😘": " emo_yeu ",
    "😭": " emo_khoc ", "😢": " emo_khoc ",
    "👍": " emo_tot ", "🔥": " emo_tot ", "👏": " emo_tot ",
    "😡": " emo_gian ", "👎": " emo_te ", "🤮": " emo_te ",
}
for e, tag in EMOJI_MAP.items():
    df = df.withColumn("text_clean", F.regexp_replace("text_clean", e, tag))
# Bỏ emoji/ký tự lạ còn lại
df = df.withColumn("text_clean", F.regexp_replace("text_clean", r"[^\p{L}\p{N}\s_]", " "))
# Giữ tín hiệu "AI" (viết hoa = trí tuệ nhân tạo, khác "ai" = who)
df = df.withColumn("text_clean", F.when(
        F.col("text").rlike(r"\bA\.?I\b"),
        F.concat(F.col("text_clean"), F.lit(" tag_ai"))
    ).otherwise(F.col("text_clean")))
# Trường hợp gõ thường: "nhạc ai", "giọng ai", "ai cover"...
df = df.withColumn("text_clean", F.regexp_replace(
        "text_clean", r"\b(nhạc|giọng|cover|voice|bằng) ai\b", "$1 tag_ai"))
df = df.withColumn("text_clean", F.regexp_replace(
        "text_clean", r"\bai (cover|voice|music)\b", "tag_ai $1"))

# 2. Teencode (bổ sung dần; cs, t, e, a chưa thêm vì còn nhập nhằng)
TEEN = {"ko": "không", "k": "không", "hok": "không", "hông": "không",
        "dc": "được", "đc": "được", "vs": "với", "wa": "quá", "wá": "quá",
        "bt": "bình thường", "mn": "mọi người", "cx": "cũng",
        "j": "gì", "z": "vậy", "b": "bạn",
        "youtobe": "youtube", "viêw": "view", "nx": "nữa", "r": "rồi",
        "ck": "chồng"}

NAMES = {"vicky nhung": "vicky_nhung", "hà anh tuấn": "hà_anh_tuấn",
         "sơn tùng": "sơn_tùng", "nhà hào môn": "nhà_hào_môn"}

def fix_teen(s):
    s = re.sub(r"\b(k{3,}|h+a+h+a+|ha(ha)+)\b", " emo_cuoi ", s)
    for n, tag in NAMES.items():
        s = s.replace(n, tag)
    s = " ".join(TEEN.get(w, w) for w in s.split())
    s = re.sub(r"\b(emo_[a-z]+)(?:\s+\1\b)+", r"\1", s)   # emo_yeu emo_yeu -> emo_yeu
    return s

@pandas_udf("string")
def tokenize(s: pd.Series) -> pd.Series:
    from pyvi import ViTokenizer
    return s.map(lambda t: ViTokenizer.tokenize(fix_teen(t)))

df = df.withColumn("text_token", tokenize("text_clean")).cache()
print("A. trước lọc 1:", df.count())
df = df.filter(F.size(F.split("text_token", " ")) >= 2)
print("B. sau lọc >=2 token:", df.count())

# 3. Cột cho GSDMM: bỏ stopword, số, từ 1 ký tự
# (đã bỏ khỏi STOP: hay, hát, bài, nghe, nhạc, ca, thích)
STOP = ["emo_yeu","emo_cuoi","emo_khoc","emo_tot","emo_gian","emo_te","là","và","của","có","này","mà","cho","đi","đã","rồi","một","những","thì",
        "lại","ra","với","còn","như","để","trong","về","nó","nào","cả","đó","từ",
        "ở","các","vào","chỉ","khi","lên","cái","ạ","nhé","nha","ơi","à","nè","ấy",
        "vì","bị","ta","cứ","sẽ","được","tôi","mình","bạn","anh","em","chị","ai",
        "không","rất","quá","lắm","thật","luôn","gì","sao","đến","phải","mới","nữa"]

df = df.withColumn(
    "text_gsdmm",
    F.array_join(
        F.filter(F.split("text_token", " "),
                 lambda w: (~w.isin(STOP)) & (~w.rlike(r"^\d+$")) & (F.length(w) >= 2)),
        " "))
df = df.filter(F.size(F.split("text_gsdmm", " ")) >= 2)
print("C. sau lọc gsdmm >=2:", df.count())

# 4. Ghi kết quả (phải nằm SAU khi df đã xử lý xong)
(df.select("comment_id", "video_id", "author", "published", "likes",
           "text", "text_token", "text_gsdmm")
   .write.mode("overwrite").parquet("/opt/data/silver/comments_tokenized"))

df.select("text_token", "text_gsdmm").show(15, truncate=80)
print("Số dòng:", df.count())

n_ai = df.filter(F.col("text_token").contains("tag_ai")).count()
print("Bình luận nhắc AI:", n_ai)