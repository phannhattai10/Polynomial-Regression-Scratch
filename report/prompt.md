# A.1 AI Prompting cho Polynomial Regression

> **Công cụ sử dụng:** Antigravity IDE với model **Claude Sonnet 4.6**
> **Phong cách:** Vibe coding — mô tả mục tiêu rõ ràng, để AI tự quyết định chi tiết cài đặt

---

## A.1.1 Prompt xây dựng chương trình

**Prompt 1:**
"Tôi đang làm đồ án học máy, mục tiêu xây dựng project Polynomial Regression trên bộ dữ liệu Auto MPG. Dữ liệu gốc là file auto-mpg.csv ở thư mục data_raw, chứa nhiều lỗi: giá trị thiếu dưới nhiều dạng khác nhau (?, NA, N/A, unknown, null, empty string), đơn vị lẫn lộn (weight chứa hậu tố 'lbs'), text viết bằng chữ cho số (cylinders = 'four'/'six'/'eight', origin = 'USA'/'Europe'/'Japan'), model year không nhất quán (70 vs 1970), dấu phẩy kiểu châu Âu trong acceleration, outlier vô lý (horsepower âm, mpg = 9999), dòng trùng lặp, dòng toàn NaN. Mục tiêu cuối: dự đoán mpg bằng Polynomial Regression từ đầu (không dùng sklearn cho luồng chính). Tạo project theo cấu trúc sau:
```
project for Polynomial Regression/
├── data_raw/
│   └── auto-mpg.csv
├── data_processed/
│   ├── train_raw.csv
│   ├── test_raw.csv
│   ├── X_train.csv
│   ├── y_train.csv
│   └── preprocessing_params.pkl
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_split.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_export_pipeline.ipynb
├── src/
│   ├── ClassPolynomial.py
│   ├── preprocessing.py
│   ├── TrainModel.py
│   └── TestScikit.py
└── report/
    ├── data_pipeline_summary.md
    └── prompt.md
```
Ràng buộc bắt buộc:
- Không dùng sklearn cho mô hình chính; không dùng train_test_split, StandardScaler, SimpleImputer, PolynomialFeatures trong luồng chính.
- Chỉ dùng NumPy, Pandas, pickle, Matplotlib.
- Split dữ liệu thô thành train/test (80/20) ngay từ đầu, trước EDA, bằng pandas + numpy thuần với np.random.seed(42).
- Test set luôn là raw, độc lập hoàn toàn và không được dùng cho fit hoặc chọn hyperparameter.
- Mọi tham số xử lý phải fit trên train_raw duy nhất: median impute, mean/std, IQR bounds, mapping giá trị, log-transform.
- EDA chỉ được chạy trên train_raw.
- Khi cần so sánh với sklearn, chỉ làm trong file TestScikit.py."

---

**Prompt 2:**
"Viết notebook 01_data_overview.ipynb. Mục tiêu: khảo sát dữ liệu thô auto-mpg.csv nhưng chưa sửa gì. Yêu cầu:
1. Đọc file bằng pandas với dtype=str để giữ nguyên các giá trị dạng chuỗi.
2. In ra head(), info(), describe().
3. In số dòng, số cột, kiểu dữ liệu từng cột.
4. Kiểm tra các dạng missing value: ?, NA, N/A, unknown, null, empty string, NaN thật.
5. Phát hiện các vấn đề dirty data sau và in ra số lượng dòng ảnh hưởng kèm 2-3 ví dụ:
   - weight có chữ 'lbs'
   - cylinders viết bằng chữ: four, six, eight
   - origin có cả số và chữ: USA/Europe/Japan và 1/2/3
   - model year không nhất quán: 70 vs 1970
   - acceleration dùng dấu phẩy kiểu châu Âu (ví dụ: '16,8')
   - dòng trùng lặp
   - dòng toàn NaN
   - outlier vô lý: horsepower âm hoặc >= 1000, mpg > 200
6. Kết thúc bằng DataFrame tổng hợp với cột: tên cột | loại vấn đề | số lượng | đề xuất xử lý sơ bộ.
7. Không sửa dữ liệu ở bước này, chỉ khảo sát và ghi nhận."

---

**Prompt 3:**
"Viết notebook 02_split.ipynb. Mục tiêu: chia data_raw/auto-mpg.csv thành 2 tập train_raw và test_raw theo tỉ lệ 80/20 bằng cách thủ công, không dùng sklearn. Yêu cầu:
1. Đọc dữ liệu thô bằng dtype=str, keep_default_na=False, na_values=[].
2. Loại bỏ các dòng toàn NaN và các dòng trùng lặp hoàn toàn trước khi split.
3. Viết hàm split_train_test(df, test_ratio=0.2, random_state=42) bằng pandas + numpy: shuffle bằng np.random.permutation rồi cắt theo tỷ lệ.
4. Lưu ra:
   - data_processed/train_raw.csv
   - data_processed/test_raw.csv
5. In ra shape của từng tập và kiểm tra không có overlap về index giữa 2 tập.
6. Thêm markdown cell giải thích vì sao cần split trước khi EDA và feature engineering để tránh data leakage.
7. Không làm sạch dữ liệu khác ngoài trùng lặp và dòng toàn NaN ở bước này."

---

**Prompt 4:**
"Viết notebook 03_eda.ipynb. Mục tiêu: thực hiện EDA CHỈ trên train_raw.csv, không được đọc test_raw. Yêu cầu:
1. Đọc file data_processed/train_raw.csv.
2. Chuẩn hoá tạm thời: chuẩn hóa missing về NaN, ép kiểu numeric cho các cột số, chuẩn hoá cylinders (chữ -> số), origin (chữ -> 1/2/3), model year (2 chữ số -> 4 chữ số) chỉ để phục vụ phân tích.
3. Vẽ histogram phân phối cho từng biến số và tính skewness. Đánh dấu biến nào có |skew| > 1.
4. Vẽ scatter plot từng biến số so với mpg và nhận xét tuyến tính hay phi tuyến.
5. Tạo correlation heatmap bằng pandas corr(). Liệt kê |r| với mpg theo thứ tự giảm dần; dùng ngưỡng: |r| < 0.3 yếu, 0.3-0.5 trung bình, >= 0.5 mạnh.
6. Vẽ boxplot cho horsepower, weight, mpg và tính ngưỡng IQR: Q1-1.5*IQR và Q3+1.5*IQR. In số lượng outlier theo ngưỡng đó.
7. Kiểm tra đa cộng tuyến: cảnh báo cặp biến có |r| > 0.8.
8. Kết luận cuối notebook: nên giữ biến nào, biến nào cần log-transform (tiêu chí |skew| > 1), nên thử degree nào trong polynomial regression.
9. Xử lý ở bước này chỉ để quan sát, không tạo tham số chính thức."

---

**Prompt 5:**
"Viết notebook 04_feature_engineering.ipynb. Mục tiêu: xây hàm xử lý dữ liệu và fit preprocessing HOÀN TOÀN chỉ trên train_raw. Yêu cầu:
1. Viết hàm clean_raw(df) xử lý toàn bộ các vấn đề dirty:
   - Ép toàn bộ cột về str rồi strip()
   - Thay thế danh sách MISSING_TOKENS (['?','NA','N/A','na','n/a','unknown','null','NULL','-','--','']) bằng np.nan
   - weight: bỏ hậu tố 'lbs', cast to numeric, giá trị < 0 -> NaN
   - acceleration: thay dấu phẩy bằng dấu chấm, cast to numeric
   - cylinders: ánh xạ chữ ('four'->4, 'six'->6, 'eight'->8, ...) hoặc cast to float
   - origin: ánh xạ ('usa'->1, 'europe'->2, 'japan'->3), numeric 1/2/3 giữ nguyên
   - model year: n < 100 -> 1900+n; ngoài khoảng 1960-2030 -> NaN
   - horsepower: < 0 hoặc >= 1000 -> NaN; mpg > 200 -> NaN
   - drop_duplicates và dropna(how='all')
2. Viết hàm fit_preprocessing(train_df) chỉ chạy trên train_raw sau khi clean_raw:
   - Tính median impute_values cho từng cột (features + target), tính trước khi điền
   - Điền tạm để tính IQR bounds cho ['horsepower','weight','mpg']: lower=Q1-1.5*IQR, upper=Q3+1.5*IQR
   - Xác định log_transform_cols dựa trên |skew| > 1 (chỉ 'horsepower' đạt tiêu chí)
   - Tính mean/std cho từng feature column SAU khi đã impute + clip + log1p
   - Trả về dict params với đầy đủ keys: features, target, log_transform_cols, impute_values, outlier_bounds, scale_mean, scale_std
3. Viết hàm transform(raw_df, params):
   - Gọi clean_raw, bỏ dòng target NaN, fillna bằng impute từ train, clip outlier theo bounds train, log1p cho log_cols, standardize (x-mean)/std
   - Trả về (X: DataFrame, y: Series)
4. Gọi fit_preprocessing trên train_raw, lưu params sang data_processed/preprocessing_params.pkl bằng pickle thuần (không dùng joblib).
5. Gọi transform(train_raw, params) để xuất X_train.csv, y_train.csv.
6. In shape của các file xuất ra. Không chạm vào test_raw."

---

**Prompt 6:**
"Viết file src/preprocessing.py. Yêu cầu:
- Trích xuất trực tiếp từ notebook 04_feature_engineering.ipynb thành module Python có thể import được.
- Khai báo các hằng số ở đầu file: MISSING_TOKENS (list), MAPPING_CYLINDERS_WORD_TO_INT (dict), MAPPING_ORIGIN_WORD_TO_INT (dict), FEATURE_COLUMNS (list 7 cột: cylinders, displacement, horsepower, weight, acceleration, model year, origin), TARGET_COLUMN = 'mpg', OUTLIER_COLUMNS = ['horsepower','weight','mpg'], LOG_TRANSFORM_COLUMNS = ['horsepower'].
- Có 3 hàm public: clean_raw(df: DataFrame) -> DataFrame, fit_preprocessing(train_df: DataFrame) -> dict, transform(raw_df: DataFrame, params: dict) -> Tuple[DataFrame, Series].
- Có các helper private: _normalize_cylinders, _normalize_origin, _normalize_model_year, _compute_iqr_bounds.
- Dùng pandas + numpy thuần, không dùng sklearn.
- Có type hint rõ ràng (from __future__ import annotations).
- Có docstring tiếng Anh cho từng hàm public, giải thích input/output và logic chính."

---

**Prompt 7:**
"Viết notebook 05_export_pipeline.ipynb. Mục tiêu: export pipeline cuối cùng và kiểm tra sanity. Yêu cầu:
1. Import clean từ src/preprocessing.py (thêm sys.path nếu cần).
2. Đọc train_raw.csv bằng dtype=str, keep_default_na=False, na_values=[].
3. Gọi fit_preprocessing() CHỈ trên train_raw, lưu params ra data_processed/preprocessing_params.pkl.
4. Gọi transform(train_raw, params) và xuất X_train.csv, y_train.csv.
5. Cell sanity check cuối: đọc lại X_train.csv/y_train.csv và kiểm tra:
   - Số lượng NaN = 0 trong X và y
   - 100% cột trong X đều là float64
   - Mean của từng cột xấp xỉ 0 (|mean| < 0.01), std xấp xỉ 1 (0.99 < std < 1.01)
6. In kết quả pass/fail rõ ràng cho từng bài kiểm tra."

---

**Prompt 8:**
"Viết file src/ClassPolynomial.py. Yêu cầu:
- Viết hàm GaussJordan(x) để tính ma trận nghịch đảo bằng phép khử Gauss-Jordan: augment ma trận [x|I], khử Gauss về dạng rút gọn, trả về phần I đã biến đổi. Xử lý trường hợp pivot = 0 bằng cách swap hàng; raise ValueError nếu ma trận suy biến.
- Viết các hàm MSE, MAE, R_2 riêng biệt nhận (y_test, y_pred) và hàm compute_metrics(y_test, y_pred) gộp trả về dict {'MSE':..., 'RMSE':..., 'MAE':..., 'R2':...}.
- Viết class Polynomial với __init__(self, degree, method='normal', learning_rate=0.1, epochs=1000):
  + method='normal': dùng normal equation W = GaussJordan(Xbar.T @ Xbar) @ Xbar.T @ y
  + method='gradient': dùng gradient descent cập nhật W theo gradient = (2/n)*(Xbar.T @ error)
  + Raise ValueError nếu method không hợp lệ
- TransformMultiFeature(x, degree): dùng itertools.combinations_with_replacement trên feature_indices để tạo tất cả tổ hợp polynomial từ degree 1 đến degree, prepend cột bias 1. Trả về Xbar shape (n, num_poly_features+1).
- Transform(x, degree): phiên bản 1D để giữ backward compat.
- Fit(Xbar, y): lưu self.W.
- Predict(Xbar): trả về Xbar @ self.W, raise ValueError nếu chưa fit.
- Viết hàm get_poly_feature_names(feature_names, degree) trả về list tên cột khớp với thứ tự của TransformMultiFeature (bao gồm '1' cho bias, tên feature, 'x^2', 'x*y', ...).
- Chỉ dùng numpy, pandas, itertools."

---

**Prompt 9:**
"Viết file src/TrainModel.py. Yêu cầu:
1. Thêm sys.stdout.reconfigure(encoding='utf-8') để hỗ trợ Unicode trên Windows.
2. Viết hàm find_project_root() đi ngược cây thư mục (tối đa 6 cấp) từ __file__ để tìm thư mục chứa cả 'src/' lẫn 'data_processed/'; raise RuntimeError nếu không tìm thấy. Lưu PROJECT_ROOT, DATA_DIR, REPORT_DIR.
3. Cấu hình matplotlib style: figure.dpi=110, savefig.dpi=300, font.size=12, axes.grid=True, bỏ top/right spine. Định nghĩa color palette: COLOR_PRIMARY='#2563eb', COLOR_SECONDARY='#f97316', COLOR_ACCENT='#16a34a', COLOR_REF_LINE='#9ca3af'.
4. Viết hàm split_train_val(X, y, val_ratio=0.2, seed=42) bằng numpy (không dùng sklearn) để tách validation riêng phục vụ chọn hyperparameter.
5. Load X_train.csv, y_train.csv. Tách sub-train/val (80/20) từ train.
6. Dùng Polynomial class duyệt degree từ 1 đến 4:
   - TransformMultiFeature trên sub-train và val
   - Fit trên sub-train bằng method='normal'
   - Compute metrics train và val
   - Lưu results = [{'degree':d, 'train':..., 'val':...}]
7. Chọn best_degree = min RMSE_val.
8. Vẽ và lưu vào report/ 4 biểu đồ: predicted vs actual (val), residual plot (val), learning curve RMSE theo degree, residual distribution histogram.
9. Sau khi chọn degree: load preprocessing_params.pkl, đọc test_raw.csv bằng dtype=str, gọi transform(test_raw, params), train lại FinalModel trên toàn bộ X_train, đánh giá trên test set.
10. Vẽ và lưu biểu đồ predicted vs actual trên test set với title ghi rõ RMSE và R²."

---

**Prompt 10:**
"Viết file src/TestScikit.py. Mục đích: đối chiếu kết quả sklearn với custom model, KHÔNG dùng trong luồng chính. Yêu cầu:
1. Dùng cùng hàm find_project_root() như TrainModel.py.
2. Dùng cùng hàm split_train_val(val_ratio=0.2, seed=42) như TrainModel.py để đảm bảo sub-train/val giống hệt.
3. Load X_train.csv, y_train.csv; tách sub-train/val.
4. Dùng sklearn.preprocessing.PolynomialFeatures(degree=DEGREE) + sklearn.linear_model.LinearRegression(fit_intercept=False) để train và predict.
5. Dùng compute_metrics từ ClassPolynomial.py để in RMSE, MAE, R² — đảm bảo con số có thể so sánh trực tiếp với TrainModel.py.
6. In W (coef_) để cross-check.
7. Cuối cùng: in số lượng cột của get_poly_feature_names() vs sklearn poly.get_feature_names_out() để xác nhận thứ tự feature khớp nhau."

---

**Prompt 11:**
"Viết file report/data_pipeline_summary.md. Yêu cầu:
- Viết bằng tiếng Việt, chuẩn báo cáo học thuật.
- Bao gồm các mục: Giới thiệu & Mục tiêu, Cấu trúc Dự án (directory tree + data flow diagram ASCII), Ràng buộc thầy yêu cầu và Giải pháp của nhóm (dạng bảng), Từ điển Dữ liệu, Nhận diện & Xử lý lỗi dirty data (8 loại lỗi), Chi tiết Preprocessing Pipeline (clean_raw, fit_preprocessing, transform), Kết quả Sanity Check, Tổng Quan Mô Hình & Đánh giá (kết quả RMSE/R² trên val và test, đối chiếu sklearn), Hướng dẫn Tái sử dụng Pipeline (code snippet Python).
- Ghi rõ kết quả cuối: Best Degree = 1, RMSE_test = 3.247, R²_test = 0.837."

---

## A.1.2 Prompt sửa lỗi và debug

**Prompt 1:**
"Script TrainModel.py báo lỗi ModuleNotFoundError: No module named 'ClassPolynomial' khi chạy từ thư mục gốc project. Hãy sửa để script tự động tìm đúng thư mục src/ và thêm vào sys.path trước khi import, tương tự cách find_project_root() đã dùng để tìm data_processed/."

---

**Prompt 2:**
"Hàm GaussJordan bị lỗi ValueError: Cannot find inverse (singular matrix) khi degree = 3 hoặc 4 với dataset này. Nguyên nhân có thể do ma trận X.T @ X gần suy biến (đa cộng tuyến cao khi mở rộng đa thức). Hãy: (1) thêm try/except trong vòng lặp degree để bắt ValueError và print thông báo SKIPPING rồi continue sang degree tiếp theo; (2) đảm bảo results chỉ chứa các degree thành công; (3) nếu results rỗng, raise RuntimeError hữu ích."

---

**Prompt 3:**
"Notebook 05_export_pipeline.ipynb báo ImportError khi import từ src/preprocessing.py vì Jupyter chạy ở thư mục notebooks/. Hãy thêm cell đầu tiên trong notebook để tự động detect project root và thêm src/ vào sys.path:

```python
import sys, os
here = os.path.abspath('')
for _ in range(6):
    if os.path.isdir(os.path.join(here, 'src')):
        sys.path.insert(0, os.path.join(here, 'src'))
        break
    here = os.path.dirname(here)
```"

---

**Prompt 4:**
"Hàm clean_raw() bị lỗi khi cột 'weight' chứa các giá trị như '2045 lbs' vì sau khi replace 'lbs' vẫn còn khoảng trắng thừa trước khi pd.to_numeric. Hãy sửa lại đoạn xử lý weight để sau khi str.replace('lbs') phải gọi thêm .str.strip() trước khi to_numeric. Và kiểm tra lại các MISSING_TOKENS có bao gồm chuỗi rỗng '' để bắt sau str.strip() không."

---

**Prompt 5:**
"Sanity check trong notebook 05 fail: mean của một số cột không xấp xỉ 0. Sau khi debug, phát hiện fit_preprocessing() tính mean/std TRƯỚC khi clip outlier và log-transform, không phải SAU. Hãy sửa thứ tự trong fit_preprocessing: (1) impute -> (2) clip IQR -> (3) log1p -> (4) tính mean/std để mean/std scale được tính trên dữ liệu đã clean hoàn toàn."

---

**Prompt 6:**
"Biểu đồ matplotlib không hiển thị khi chạy TrainModel.py từ terminal (không có GUI). Hãy thêm import matplotlib; matplotlib.use('Agg') TRƯỚC khi import matplotlib.pyplot, và thay plt.show() bằng hàm save_fig() lưu ra report/. Đảm bảo import matplotlib trước matplotlib.pyplot ở đầu file."

---

**Prompt 7:**
"Khi đọc test_raw.csv trong TrainModel.py bằng pd.read_csv thông thường, pandas tự động parse '?' thành NaN nhưng cũng tự parse một số chuỗi hợp lệ không mong muốn. Hãy đọc bằng pd.read_csv(..., dtype=str, keep_default_na=False, na_values=[]) để giữ nguyên toàn bộ giá trị thô, sau đó để hàm clean_raw() trong preprocessing.py xử lý tất cả — đây là cách đảm bảo pipeline nhất quán giữa train và test."

---

**Prompt 8:**
"TestScikit.py dùng split_train_val với seed=42 nhưng kết quả metric khác với TrainModel.py. Kiểm tra lại: cả hai file có dùng cùng val_ratio=0.2 và cùng thứ tự shuffled_idx[:n_val] là val, shuffled_idx[n_val:] là train không? Hãy đồng bộ hàm split_train_val() giữa hai file để đảm bảo sub-train và val giống hệt nhau."

---

## A.1.3 Prompt cải thiện và tối ưu mã nguồn

**Prompt 1:**
"Hàm TransformMultiFeature trong ClassPolynomial.py hiện dùng vòng lặp Python thuần để nhân từng term với vòng for feature_index in combo. Hãy tối ưu bằng cách dùng np.prod() hoặc np.multiply.reduce() trên các cột được chọn. Đảm bảo kết quả numerically identical với phiên bản cũ."

---

**Prompt 2:**
"Cập nhật TrainModel.py để cấu hình matplotlib style tập trung bằng plt.rcParams.update({...}) ở đầu file thay vì set style trong từng hàm vẽ riêng lẻ. Định nghĩa bộ màu consistent: COLOR_PRIMARY='#2563eb' (blue), COLOR_SECONDARY='#f97316' (orange), COLOR_ACCENT='#16a34a' (green), COLOR_REF_LINE='#9ca3af' (gray). Dùng edgecolors='white', linewidths=0.6 cho scatter plot để trông đẹp hơn. Bỏ top/right spine trên tất cả biểu đồ."

---

**Prompt 3:**
"Thêm hàm gradient descent vào class Polynomial (method='gradient') như một phương án thay thế cho normal equation. Yêu cầu: khởi tạo self.W = np.zeros(Xbar.shape[1]), lặp epochs lần với gradient = (2/n) * Xbar.T @ (Xbar @ W - y), W = W - learning_rate * gradient. Giữ method='normal' là default. Đây là để minh họa cả hai phương pháp trong báo cáo."

---

**Prompt 4:**
"Thêm vào TrainModel.py phần in bảng SUMMARY sau khi duyệt xong tất cả degree, hiển thị: degree | RMSE_train | RMSE_val | R2_val theo dạng căn chỉnh f-string. Đánh dấu best degree bằng dấu => ở đầu dòng. Sau đó thêm phần cross-reference: dùng get_poly_feature_names(feature_names, best_degree) và in số lượng cột sinh ra để xác nhận khớp với Xbar.shape[1]."

---

**Prompt 5:**
"Refactor preprocessing.py để tách logic normalize thành các hàm private nhỏ: _normalize_cylinders(value), _normalize_origin(value), _normalize_model_year(value), _compute_iqr_bounds(series). Mỗi hàm xử lý đúng một trường hợp, có type hint, dễ unit test độc lập. Hàm clean_raw() chỉ cần gọi df['cylinders'].apply(_normalize_cylinders) v.v. thay vì viết logic inline."

---

**Prompt 6:**
"Cải thiện robustness của hàm find_project_root() trong TrainModel.py và TestScikit.py: thêm tham số required_dirs=('src', 'data_processed') và max_levels=6. Nếu không tìm được root sau max_levels bước, raise RuntimeError với thông báo hướng dẫn người dùng chạy 05_export_pipeline.ipynb trước. Dùng hàm này ở đầu cả hai script."

---

**Prompt 7:**
"Thêm cell markdown vào notebook 03_eda.ipynb phần kết luận để liệt kê rõ ràng: (1) các biến có |corr với mpg| >= 0.5 (mạnh), (2) biến nào có |skew| > 1 cần log-transform (chỉ horsepower đạt ngưỡng trên dataset này), (3) cặp biến nào có |r| > 0.8 (đa cộng tuyến: cylinders-displacement, cylinders-weight, displacement-weight), (4) khuyến nghị thử degree 1-4 và chọn theo RMSE_val. Đây là tài liệu hóa quyết định EDA để giải thích cho preprocessing pipeline."

---

**Prompt 8:**
"Sửa lại toàn bộ split strategy theo đúng yêu cầu thầy: split thành train/test (80/20) trước tất cả (notebook 02), chỉ tách val từ train bên trong TrainModel.py ở bước chọn hyperparameter. Xóa val_raw.csv ra khỏi data_processed/ nếu đã tạo. Cập nhật data_pipeline_summary.md để phản ánh đúng sơ đồ luồng: auto-mpg.csv -> (02_split) -> train_raw (80%) + test_raw (20%) -> (preprocessing fit trên train) -> X_train, y_train -> (TrainModel tách val từ train) -> sub-train (80%) + val (20%) -> chọn degree -> Final Model -> test."
