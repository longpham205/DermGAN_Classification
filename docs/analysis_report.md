# Analysis Module Report

---

# 1. Tổng quan hệ thống phân tích (Analysis Module)

Module `analysis/` được thiết kế như một **research-grade evaluation framework** nhằm:

- Đánh giá hiệu năng tổng thể của mô hình  
- Phân tích hiệu năng theo từng lớp bệnh  
- So sánh nhiều experiment  
- Kiểm định ý nghĩa thống kê  
- Phân tích không gian biểu diễn (representation space)  
- (Tùy chọn) Phân tích lâm sàng (clinical interpretability)  

Mục tiêu không chỉ dừng ở việc báo cáo accuracy, mà xây dựng một pipeline phân tích toàn diện phục vụ:

- So sánh baseline vs augmentation  
- Đánh giá GAN augmentation  
- Kiểm định statistical significance  
- Chuẩn bị kết quả cho publication  

---

# 2. Kiến trúc tổng thể của Analysis Pipeline

## Cấu trúc thư mục

```
src/
└── analysis
├── clinical
├── core
├── performance
├── representation
└── statistics
```

## Pipeline tổng quát

```
Experiment Outputs
(metrics.json, predictions.csv, embeddings.npy)
↓
load_results.py
↓
Performance Analysis
↓
Statistical Testing
↓
(Optionally) Representation Analysis
↓
(Optionally) Clinical / Demographic Analysis
```


---

# 3. Core Layer – Nền tảng của hệ thống

Thư mục: `analysis/core/`

## 3.1 load_results.py

### Mục đích

- Load toàn bộ thông tin của một experiment  
- Chuẩn hóa dữ liệu đầu vào cho các module phía sau  

### Output chính: `ExperimentResult`

Chứa:

- `exp_name`  
- `metrics`  
- `classification_report`  
- `confusion_matrix`  
- `predictions_path`  
- `exp_dir`  

### Vai trò

Tách biệt hoàn toàn:

- File system  
- Logic phân tích  

Đây là lớp abstraction quan trọng giúp toàn bộ pipeline sạch và mở rộng được.

---

## 3.2 compare_experiments.py

### Mục đích

So sánh nhiều experiment trên cùng bộ metric.

### Tính năng

- Tổng hợp accuracy  
- Macro F1  
- Weighted F1  
- Macro AUC  
- Xuất bảng CSV so sánh  

### Ý nghĩa

Giúp:

- So sánh baseline vs augmentation  
- Đánh giá tác động của GAN  
- Tạo bảng kết quả cho paper  

---

## 3.3 metrics_utils.py

### Vai trò

- Các utility tính toán metric  
- Chuẩn hóa cách tính để đảm bảo consistency  

---

# 4. Performance Analysis Layer

Thư mục: `analysis/performance/`

Đây là lớp đánh giá mô hình ở mức prediction.

---

## 4.1 overall_analysis.py

### Phân tích

- Accuracy  
- Macro F1  
- AUC  
- Balanced accuracy  

### Ý nghĩa

Đánh giá tổng quát mô hình trên toàn bộ dataset.

---

## 4.2 per_class_analysis.py

### Phân tích

- Precision theo từng lớp  
- Recall theo từng lớp  
- F1-score theo từng lớp  

### Ý nghĩa

Rất quan trọng với HAM10000 vì:

- Dataset mất cân bằng  
- Melanoma là minority class  

Giúp đánh giá:

- Model có bỏ sót melanoma không?  
- Có bias với lớp phổ biến không?  

---

## 4.3 confusion_analysis.py

### Phân tích

- Ma trận nhầm lẫn  
- Pattern nhầm lẫn giữa các lớp  

### Ý nghĩa

Ví dụ:

- Model nhầm melanoma thành nevus?  
- Hay nhầm BCC thành AKIEC?  

Cực kỳ quan trọng trong nghiên cứu y tế.

---

## 4.4 calibration_analysis.py

### Phân tích

- Calibration curve  
- Expected Calibration Error (ECE)  

### Ý nghĩa

Trong y khoa, xác suất dự đoán phải đáng tin cậy.

Model có confidence cao nhưng sai → nguy hiểm.

---

# 5. Statistics Layer – Kiểm định ý nghĩa thống kê

Thư mục: `analysis/statistics/`

Đây là phần nâng cấp pipeline lên mức publication.

---

## 5.1 bootstrap_ci.py

### Mục đích

Tính confidence interval cho:

- Accuracy  
- Macro F1  
- Macro AUC  

### Phương pháp

Bootstrap resampling:

- Lấy mẫu có hoàn lại  
- Lặp N lần (mặc định 1000)  
- Tính phân vị 2.5% – 97.5%  

### Ý nghĩa

Không chỉ báo cáo:

```
Accuracy = 0.87
```


mà báo cáo:

```
Accuracy = 0.87 (95% CI: 0.84 – 0.90)
```


→ Chuẩn publication.

---

## 5.2 mcnemar_test.py

### Mục đích

So sánh 2 mô hình trên cùng test set.

### Phù hợp khi

- So sánh baseline vs augmented  
- So sánh ResNet vs EfficientNet  

### Kiểm định

Null hypothesis: 2 model có performance tương đương  

Nếu p-value < 0.05 → cải thiện có ý nghĩa thống kê.

---

## 5.3 significance_report.py

### Tổng hợp

- McNemar test  
- Bootstrap CI  
- So sánh metric  
- Xuất báo cáo thống kê  

### Vai trò

Đây là module chính biến pipeline thành:

**Research-grade evaluation framework.**

---

# 6. Representation Analysis Layer

Thư mục: `analysis/representation/`

---

## 6.1 embedding_analysis.py

### Phân tích

- Khoảng cách intra-class  
- Khoảng cách inter-class  
- Compactness  
- Separability  

### Ý nghĩa

Kiểm tra:

GAN augmentation có giúp:

- Tăng separation giữa lớp?  
- Làm feature space rõ ràng hơn?  

Đây là phân tích ở mức representation, không chỉ prediction.

---

## 6.2 feature_space_analysis.py

Phân tích sâu hơn về:

- Cấu trúc không gian đặc trưng  
- Clustering behavior  
- Overlap giữa các lớp  

---

# 7. Clinical Layer

Thư mục: `analysis/clinical/`

---

## 7.1 abcd_analysis.py

### Mục tiêu

Phân tích sai số theo quy tắc ABCD:

- Asymmetry  
- Border  
- Color  
- Diameter  

### Lưu ý quan trọng

Dataset HAM10000 không có ABCD annotation.

Vì vậy phần này:

- Có trong pipeline  
- Nhưng không dùng được với HAM10000 mặc định  

Nếu muốn dùng cần:

- Dataset có ABCD features  
- Hoặc tự trích xuất từ segmentation  

---

# 8. Tóm tắt nhiệm vụ từng nhóm file

| Layer           | Mục tiêu                              |
|---------------|----------------------------------------|
| core          | Load và chuẩn hóa experiment           |
| performance   | Đánh giá prediction                   |
| statistics    | Kiểm định ý nghĩa                     |
| representation| Phân tích feature space               |
| clinical      | Phân tích theo đặc trưng lâm sàng     |

---

# 9. Kết quả mà hệ thống tạo ra

Pipeline tạo ra:

- `metrics.json`  
- `experiment_comparison.csv`  
- `confusion_matrix.json`  
- Bootstrap CI report  
- McNemar test result  
- `embedding_analysis.json`  
- (Optional) clinical report  

Đủ để:

- Viết bảng kết quả cho paper  
- Viết phần statistical validation  
- Chứng minh GAN augmentation có ý nghĩa  

---

# 10. Ý nghĩa khoa học của hệ thống

Hệ thống này cho phép:

- So sánh nhiều experiment một cách hệ thống  
- Kiểm định statistical significance  
- Phân tích sâu feature space  
- Đảm bảo reproducibility  
- Chuẩn bị output sẵn sàng cho publication  

Nó vượt xa việc chỉ in accuracy.

---

# 11. Kết luận phần Analysis

Module Analysis trong hệ thống DermGAN:

- Được thiết kế theo hướng modular  
- Tách biệt rõ layer  
- Không phụ thuộc lẫn nhau  
- Mở rộng được  
- Đủ tiêu chuẩn cho nghiên cứu khoa học  

Với HAM10000, phần quan trọng nhất là:

- Overall performance  
- Per-class analysis  
- Confusion matrix  
- Bootstrap CI  
- McNemar significance test  
- Representation analysis  

## 12. Kết luận chung

Phần Analysis không chỉ nhằm báo cáo kết quả, mà:

- Là công cụ kiểm tra độ tin cậy và tính lâm sàng của hệ thống GAN-based classification.

- Nó giúp chuyển hệ thống từ:

"Model có accuracy cao"

thành:

"Model được kiểm chứng, hiểu rõ hành vi, và có thể đánh giá mức độ tin cậy trong bối cảnh y khoa."

