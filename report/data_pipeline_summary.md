# BÁO CÁO TỔNG QUAN XỬ LÝ DỮ LIỆU & PIPELINE (DATA PIPELINE SUMMARY)

---

## 1. Giới thiệu & Mục tiêu Dự án (Project Overview & Objectives)

- **Tập dữ liệu:** Auto MPG (`auto-mpg.csv`) chứa các thông số kỹ thuật xe hơi và mức tiêu thụ nhiên liệu (`mpg`).
- **Mục tiêu:** Dự đoán mức tiêu thụ nhiên liệu `mpg` (miles per gallon) dựa trên 7 đặc trưng kỹ thuật số.
- **Mô hình triển khai:** Thuật toán Regression Đa thức (Polynomial Regression) tự viết từ đầu (from scratch) sử dụng Python, NumPy và Pandas; luồng huấn luyện chính không dùng `scikit-learn`. Việc dùng `scikit-learn` chỉ được giới hạn ở mục đích đối chiếu/so sánh với kết quả của mô hình tự xây dựng.
- **Nguyên tắc chống rò rỉ dữ liệu (Data Leakage Prevention):**
  - Thực hiện chia dữ liệu Train/Test (80/20) **trước** khi thực hiện bất kỳ phân tích EDA hay học tham số xử lý nào.
  - Hàm `fit_preprocessing()` chỉ được gọi **duy nhất 1 lần trên `train_raw`**.
  - Tập `test_raw` luôn được giữ ở trạng thái thô độc lập hoàn toàn, không tham gia vào fit tham số hay chọn mô hình.
  - Khi cần tập validation ở bước evaluation để lựa chọn siêu tham số (chọn bậc đa thức), tập val sẽ được tách trực tiếp từ tập `train`.

### 1.1 Ràng buộc buổi đầu thầy yêu cầu và giải pháp của nhóm

- **Ràng buộc buổi đầu thầy yêu cầu:**
  - **Không sử dụng thư viện scikit-learn** cho phần cài đặt thuật toán chính.
  - **Tập test thực tế luôn là tập thô** — không được tiền xử lý hay tham gia vào quá trình fit thông số.
  - **Sử dụng các kỹ thuật chống rò rỉ dữ liệu (data leakage prevention)** trong toàn bộ pipeline.

- **Giải pháp của nhóm đề ra:**

| Ràng buộc | Giải pháp |
| :--- | :--- |
| Không dùng scikit-learn | Chỉ sử dụng `scikit-learn` để **so sánh kết quả** . Toàn bộ thuật toán, pipeline tiền xử lý được cài đặt thuần NumPy/Pandas. |
| Tập test thực tế là tập thô | **Split tập thô đầu tiên** (Notebook 01) để chia 2 tập train/test ở dạng thô. Tập test được giữ độc lập hoàn toàn. Nếu cần tập val ở evaluation để chọn hyperparameters thì split val từ train. |
| Chống rò rỉ dữ liệu | **Chỉ fit tham số duy nhất trên tập train**, xuất ra file `.pkl` để có thể tái sử dụng biến đổi dữ liệu mới trong tương lai một cách nhất quán. Val và test tuyệt đối không tham gia |

---

## 2. Cấu trúc Dự án & Luồng Dữ liệu (Project Structure & Data Flow)

### 2.1 Sơ đồ thư mục dự án (Directory Structure)
```
project for Polynomial Regression/
├── data_raw/
│   └── auto-mpg.csv                   # Dữ liệu thô ban đầu (chứa nhiều nhiễu, lỗi)
├── data_processed/
│   ├── train_raw.csv, test_raw.csv    # Dữ liệu thô đã phân chia (80/20)
│   ├── preprocessing_params.pkl       # Bộ tham số tiền xử lý được fit từ train_raw
│   └── X_train.csv, y_train.csv       # Dữ liệu Train đã làm sạch & chuẩn hóa
├── notebooks/
│   ├── 01_data_overview.ipynb         # Khảo sát dữ liệu thô ban đầu
│   ├── 02_split.ipynb                 # Chia Train/Test (80/20) thuần Pandas + NumPy
│   ├── 03_eda.ipynb                   # Phân tích khám phá dữ liệu (EDA) trên train_raw
│   ├── 04_feature_engineering.ipynb   # Xây dựng & thử nghiệm pipeline xử lý dữ liệu
│   └── 05_export_pipeline.ipynb       # Xuất pipeline, fit train & kiểm tra tính hợp lệ
├── src/
│   ├── preprocessing.py               # Thư viện tiền xử lý (clean_raw, fit, transform)
│   ├── ClassPolynomial.py             # Lớp Polynomial tự định nghĩa & Gauss-Jordan
│   ├── TrainModel.py                  # Script tách val từ train, chọn degree, train & test
│   └── TestScikit.py                  # Script đối chiếu kết quả với scikit-learn
└── report/
    ├── 01_missing_values_rate.png đến 10_pairplot_top3_vars.png   # Đồ thị EDA
    ├── 11_predicted_vs_actual.png đến 15_final_test_predicted_vs_actual.png # Đồ thị Model
    ├── promt.md                       # Nhật ký yêu cầu đồ án
    └── data_pipeline_summary.md       # Báo cáo tổng hợp luồng xử lý
```

### 2.2 Sơ đồ luồng dữ liệu (Data Pipeline Architecture)
```
auto-mpg.csv
       │
       ▼ (02_split.ipynb: Lọc dòng rỗng & dòng trùng lặp, np.random.seed(42))
 ┌─────┴─────────────────────────────────────┐
 │ Train Raw (80% ~ 322 mẫu)                 │ Test Raw (20% ~ 81 mẫu)
 └─────┬─────────────────────────────────────┘ (Giữ nguyên thô độc lập hoàn toàn)
       │
       ▼ (clean_raw: Chuẩn hóa định dạng, kiểu dữ liệu, đại diện NaN/số)
 Train Clean (320 mẫu)
       │
       ▼ (fit_preprocessing: Học Median, Bound IQR, Log1p transform, Mean & Std)
 preprocessing_params.pkl
       │
       ▼ (transform: Impute, Clip Outliers, Log1p, Standardize)
 X_train.csv (315, 7) & y_train.csv (315,)
       │
       ▼ (src/TrainModel.py: Tách Validation từ Train cho bước Evaluation)
 ┌─────┴─────────────────────────┐
 │ Sub-Train (80% ~ 252 mẫu)     │ Validation (20% ~ 63 mẫu)
 └─────┬─────────────────────────┴─────────────────────────┘
       ▼
 Đánh giá Degree 1..4 trên tập Validation -> Chọn Degree tối ưu
       │
       ▼ Huấn luyện lại Final Model trên toàn bộ Train (100% ~ 315 mẫu)
 Final Model (Degree = 1)
       │
       ▼ Đánh giá duy nhất 1 lần trên Test Raw (được transform qua params.pkl)
 Kết quả Đánh giá Thực tế trên Tập Test (RMSE = 3.247, R² = 0.837)
```

---

## 3. Từ điển Dữ liệu & Nhận diện Lỗi Dữ liệu thô (Data Dictionary & Dirty Issues)

### 3.1 Từ điển Dữ liệu (Data Dictionary)
| Tên cột | Ý nghĩa | Kiểu mong đợi | Đơn vị | Khoảng hợp lệ mong đợi |
| :--- | :--- | :--- | :--- | :--- |
| `mpg` | Mức tiêu thụ nhiên liệu (Miles per gallon) - **TARGET** | Float | mpg | ~9 - 47 |
| `cylinders` | Số xilanh động cơ | Integer | count | {3, 4, 5, 6, 8} |
| `displacement` | Dung tích xilanh động cơ | Float | cubic inch | ~68 - 455 |
| `horsepower` | Công suất động cơ | Float | hp | ~46 - 230 |
| `weight` | Trọng lượng xe | Float | lbs | ~1613 - 5140 |
| `acceleration` | Thời gian tăng tốc từ 0-60 mph | Float | seconds | ~8 - 25 |
| `model year` | Năm sản xuất (2 chữ số, ví dụ 70 = 1970) | Integer | year | 70 - 82 (1970 - 1982) |
| `origin` | Nguồn gốc xuất xứ (1=USA, 2=Europe, 3=Japan) | Integer/Categorical | code | {1, 2, 3} |
| `car name` | Tên mẫu xe (Identifier, không dùng để dự đoán) | String | - | - |

### 3.2 Nhận diện & Xử lý các dạng dữ liệu lỗi (Dirty Issues Identified)
1. **Giá trị khuyết ẩn dưới ký tự đặc biệt:** Ký tự như `?`, `NA`, `N/A`, `unknown`, `null`, `-`, `--`, chuỗi rỗng `""`.
2. **Đơn vị lẫn lộn trong chuỗi:** Cột `weight` chứa hậu tố đơn vị `"lbs"` (ví dụ: `"2045 lbs"`).
3. **Số ghi bằng chữ (Word-written numbers):**
   - Cột `cylinders` ghi dạng từ: `"four"`, `"six"`, `"eight"`.
   - Cột `origin` ghi dạng chuỗi quốc gia: `"USA"`, `"Europe"`, `"Japan"`.
4. **Dấu phân cách thập phân kiểu Châu Âu:** Cột `acceleration` dùng dấu phẩy `,` thay vì dấu chấm `.` (ví dụ: `"16,8"`).
5. **Giá trị âm hoặc phi thực tế:**
   - Cột `weight` có giá trị âm (ví dụ: `-1945`).
   - Cột `horsepower` có giá trị âm hoặc vượt quá $1000$ hp.
   - Cột `model year` nằm ngoài khoảng 1960–2030.
   - Cột `mpg` vượt quá 200 mpg (ví dụ: 9999).
6. **Định dạng năm không thống nhất:** Kết hợp 2 chữ số (70..82) và 4 chữ số (1970..1982).
7. **Khoảng trắng thừa & chữ hoa/thường:** Tên xe và các trường văn bản chứa khoảng trắng đầu/cuối.
8. **Dòng rỗng & dòng trùng lặp:** Dòng không có dữ liệu hoặc trùng lặp toàn bộ thông tin.

---

## 4. Chi tiết Các Bước Tiền Xử Lý Dữ Liệu (Detailed Preprocessing Pipeline)

Hàm tiền xử lý chính được định nghĩa trong [src/preprocessing.py]:

### 4.1 Hàm `clean_raw(df)` - Chuẩn hóa Định dạng & Kiểu Dữ liệu
- **Chuyển đổi chuỗi & cắt khoảng trắng:** Ép kiểu toàn bộ các cột về `str` và `.str.strip()`.
- **Nhận diện giá trị khuyết:** Thay thế danh sách `MISSING_TOKENS` bằng `np.nan`.
- **Làm sạch cột `weight`:** Loại bỏ chuỗi `"lbs"`, chuyển sang dạng số, chuyển các giá trị $< 0$ thành `np.nan`.
- **Chuyển đổi dấu thập phân:** Thay `,` bằng `.` ở cột `acceleration`.
- **Chuẩn hóa giá trị dị biệt:** Chuyển các giá trị không hợp lệ của `horsepower` ($< 0$ hoặc $\ge 1000$), `mpg` ($> 200$), `model year` ($< 1960$ hoặc $> 2030$) thành `np.nan`.
- **Ánh xạ chữ số:**
  - `cylinders`: `"four"` $ightarrow 4$, `"six"` $ightarrow 6$, v.v.
  - `origin`: `"usa"`/`"us"` $ightarrow 1$, `"europe"`/`"eu"` $ightarrow 2$, `"japan"`/`"jp"` $ightarrow 3$.
- **Chuẩn hóa năm:** Chuyển năm 2 chữ số $n < 100 ightarrow 1900 + n$.
- **Loại bỏ dòng rỗng & trùng lặp:** Loại bỏ hoàn toàn dòng trùng và dòng toàn `NaN`.

### 4.2 Hàm `fit_preprocessing(train_df)` - Học Tham Số Xử Lý (Fit on Train Only)
Hàm này **chỉ chạy duy nhất trên `train_clean`** để tính toán các tham số:
1. **Xử lý thiếu (Imputation):** Tính **Median** của từng cột trên tập Train (trước khi điền giá trị khuyết).
2. **Ngưỡng ngoại lệ IQR (Outlier Bounds):** Tính khoảng liên tứ phân vị cho 3 cột `['horsepower', 'weight', 'mpg']`:
   $$Q_1 = \text{Quantile}(0.25), \quad Q_3 = \text{Quantile}(0.75), \quad \text{IQR} = Q_3 - Q_1$$
   $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
3. **Biến đổi Log-transform:** Áp dụng hàm $\log(1 + x)$ (`np.log1p`) cho đặc trưng có độ lệch skewness cao ($|\text{skew}| > 1.0$) - chỉ có `horsepower` thoả mãn tiêu chí này.
4. **Chuẩn hóa Z-Score Scaling (Mean & Std):** Tính giá trị trung bình $\mu$ và độ lệch chuẩn $\sigma$ trên tập Train (sau khi đã Impute, Clip Outliers và Log-transform):
   $$\mu = \text{mean}(X_{\text{train}}), \quad \sigma = \text{std}(X_{\text{train}})$$

Tất cả các tham số được lưu vào file `data_processed/preprocessing_params.pkl`.

### 4.3 Hàm `transform(raw_df, params)` - Chuyển đổi Dữ liệu
Sử dụng bộ tham số `params` đã fit từ tập Train để biến đổi dữ liệu thô:
1. Gọi `clean_raw(raw_df)`.
2. Loại bỏ các dòng có mục tiêu `mpg` bị thiếu (không tiến hành điền nhãn `mpg`).
3. Điền giá trị khuyết ở các đặc trưng đầu vào bằng `impute_values` (Median từ Train).
4. Cắt tỉa giá trị ngoại lệ theo `outlier_bounds` (IQR từ Train).
5. Áp dụng biến đổi `log1p` cho đặc trưng `horsepower`.
6. Chuẩn hóa đặc trưng Z-Score:
   $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
7. Trả về ma trận đặc trưng $X$ (7 cột) và chuỗi mục tiêu $y$ (`mpg`).

---

## 5. Kết quả Sanity Check & Kiểm Tra Định Dạng (Sanity Check Results)

Sau khi chạy [notebooks/05_export_pipeline.ipynb], toàn bộ các bài kiểm tra chất lượng file xuất trong `data_processed/` đều đạt kết quả xuất sắc:

1. **Kiểm tra giá trị `NaN`:** $0$ giá trị `NaN` trong `X_train.csv`, `y_train.csv`.
2. **Kiểm tra kiểu dữ liệu (Data Dtypes):** $100\%$ các cột trong ma trận $X$ đều là `float64`, không còn bất kỳ chuỗi văn bản nào.
3. **Kiểm tra tính chuẩn hóa (Z-score Check on Train):**
   - Giá trị trung bình $\mu$ của từng cột xấp xỉ $0$.
   - Độ lệch chuẩn $\sigma$ của từng cột xấp xỉ $1$.

---

## 6. Tổng Quan Mô Hình Polynomial Regression & Đánh giá (Model & Evaluation Summary)

Mô hình được phát triển trong [src/ClassPolynomial.py] bao gồm:
- **Biến đổi đa thức đa biến (Multi-feature Polynomial Expansion):** Tự động tạo các tổ hợp đa thức từ degree $1$ đến $d$ bằng `itertools.combinations_with_replacement`.
- **Giải hệ phương trình tuyến tính ngầm:** Sử dụng phép khử Gauss-Jordan (`GaussJordan`) nghịch đảo ma trận $(X^T X)^{-1}$ tìm trọng số $W = (X^T X)^{-1} X^T y$.
- **Huấn luyện & Chọn Bậc Đa Thức (Degree Selection):** Chạy [src/TrainModel.py]:
  - Tách một phần từ tập Train làm tập Validation (ví dụ 80% Sub-train / 20% Validation) để đánh giá các bậc $d = 1, 2, 3, 4$.
  - Kết quả được lựa chọn dựa trên chỉ số **RMSE trên tập Validation** để tránh hiện tượng quá khớp (Overfitting). Kết quả chọn được **Degree = 1** (Validation RMSE = 3.771).
  - Sau khi chọn bậc tối ưu, huấn luyện lại Final Model trên toàn bộ tập Train (315 mẫu).
  - Đánh giá duy nhất 1 lần trên tập Test thô `test_raw.csv` (được transform bằng `preprocessing_params.pkl`): **RMSE = 3.247, R² = 0.837**, khẳng định khả năng tổng quát hóa xuất sắc.
  - Đối chiếu độ chính xác thu được với thư viện `scikit-learn` qua [src/TestScikit.py], kết quả các chỉ số sai số trùng khớp tuyệt đối đến từng chữ số thập phân.

---

## 7. Hướng dẫn Tái sử dụng Pipeline (How to Use Pipeline)

Để biến đổi một tập dữ liệu thô mới (ví dụ `test_raw.csv`):
```python
import os
import pickle
import pandas as pd
from src.preprocessing import transform

# 1. Đọc bộ tham số đã fit từ tập Train
with open('data_processed/preprocessing_params.pkl', 'rb') as f:
    params = pickle.load(f)

# 2. Đọc dữ liệu test thô
test_raw = pd.read_csv('data_processed/test_raw.csv', dtype=str, keep_default_na=False, na_values=[])

# 3. Tiến hành transform (KHÔNG refit)
X_test, y_test = transform(test_raw, params)
```

---
