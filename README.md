# GSDMM và Sentiment

Bộ chạy hoàn chỉnh nằm trong thư mục mới **[gsdmm_sentiment/](gsdmm_sentiment/)**, tách riêng khỏi thư mục `test/` đã có.

```text
GSDMMnSentiment/
├── README.md
├── gsdmm_sentiment/     # Bộ chạy GSDMM → một khía cạnh → sentiment
│   ├── README.md        # Hướng dẫn từng lệnh PowerShell và Docker
│   ├── compose.yaml
│   ├── jobs/
│   ├── config/
│   ├── requirements/
│   ├── tests/
│   └── data/
└── test/                # Giữ nguyên code và dữ liệu đã tải lên
```

**Mỗi bình luận có một khía cạnh chính**, kế thừa từ cụm GSDMM. Sentiment được thêm cho chính bình luận đó, không nhân dòng để gán nhiều khía cạnh.

Mở [hướng dẫn chạy chi tiết](gsdmm_sentiment/README.md). Sau khi clone repo, chuyển vào thư mục bộ chạy trước khi thực hiện các lệnh Docker:

```powershell
cd .\GSDMMnSentiment\gsdmm_sentiment
```

Kết quả kiểm tra được ghi tại [VALIDATION.md](gsdmm_sentiment/VALIDATION.md).
