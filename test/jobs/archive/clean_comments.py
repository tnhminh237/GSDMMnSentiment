from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("ai-music-comments").getOrCreate()

# Đọc CSV: bắt buộc multiLine vì bình luận có xuống dòng trong ngoặc kép
df = (spark.read
      .option("header", True)
      .option("multiLine", True)
      .option("quote", '"')
      .option("escape", '"')
      .csv("/opt/data/raw/comments.csv"))

print("Số dòng ban đầu:", df.count())

# 1. Kiểu dữ liệu
df = (df.withColumn("published", F.to_timestamp("published"))
        .withColumn("likes", F.col("likes").cast("int"))
        .fillna({"likes": 0}))

# 2. Bỏ trùng, bỏ rỗng
df = df.dropDuplicates(["comment_id"]).filter(F.col("text").isNotNull())

# 3. Chuẩn hóa văn bản
df = (df.withColumn("text_clean", F.lower(F.col("text")))
        .withColumn("text_clean", F.regexp_replace("text_clean", r"https?://\S+", " "))
        .withColumn("text_clean", F.regexp_replace("text_clean", r"\b(k{3,}|h{3,}|(ha){2,})\b", " emo_cuoi "))
        .withColumn("text_clean", F.regexp_replace("text_clean", r"(.)\1{2,}", "$1"))
        .withColumn("text_clean", F.regexp_replace("text_clean", r"\s+", " "))
        .withColumn("text_clean", F.trim("text_clean")))

# 4. Lọc tiếng Việt bằng ký tự có dấu
# Ký tự chỉ có trong tiếng Việt, hoặc từ tiếng Việt phổ biến
VI_ONLY = r"[ăơưđạảãấầẩẫậắằẳẵặẹẻẽếềểễệịỉĩọỏõốồổỗộớờởỡợụủũứừửữựỳỵỷỹ]"
VI_WORDS = r"\b(không|của|và|là|này|rất|quá|hay|bài|nghe|giọng|nhạc|được|cho|với|nhưng|anh|em|chị)\b"
df = df.withColumn("is_vi", F.col("text_clean").rlike(VI_ONLY) | F.col("text_clean").rlike(VI_WORDS))

# Thống kê theo video (thay cho đoạn groupBy cũ)
stats = df.groupBy("video_id").agg(
    F.count("*").alias("so_binh_luan"),
    F.sum(F.col("is_vi").cast("int")).alias("so_binh_luan_vi"),
    F.avg(F.col("is_vi").cast("int")).alias("vi_ratio")
)
print("Tổng số video:", stats.count())
stats.orderBy(F.desc("so_binh_luan_vi")).show(50, truncate=False)

# Giữ bình luận tiếng Việt, từ 3 từ trở lên
df_vi = df.filter(F.col("is_vi") & (F.size(F.split("text_clean", " ")) >= 3))
df_vi = df_vi.withColumn("n_words", F.size(F.split("text_clean", " ")))
long_c = df_vi.filter("n_words >= 8").dropDuplicates(["text_clean"])
short_c = df_vi.filter("n_words < 8")
df_vi = short_c.unionByName(long_c).drop("n_words")
print("Số dòng sau khi lọc:", df_vi.count())

df_vi.write.mode("overwrite").parquet("/opt/data/silver/comments_clean")

# them (co the bo)
ai_vids = (spark.read.option("header", True).csv("/opt/data/gold/video_ai_final.csv")
           .filter(F.col("ai_final") == "True").select("video_id"))
d = df.join(ai_vids, "video_id")
print("RESULT ai_total:", d.count())
print("RESULT ai_no_accent:", d.filter(~F.col("is_vi")).count())
print("RESULT ai_accent_lt3words:",
      d.filter(F.col("is_vi") & (F.size(F.split("text_clean", " ")) < 3)).count())

(d.filter(~F.col("is_vi")).select("text").sample(0.06, seed=1).limit(300)
   .toPandas().to_csv("/opt/data/gold/sample_no_accent.csv", index=False, encoding="utf-8-sig"))