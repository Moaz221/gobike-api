from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import math

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def safe_float(val):
    """تحويل آمن لأي قيمة إلى float، مع معالجة NaN و Infinity"""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 4)
    except (ValueError, TypeError):
        return None


def safe_int(val):
    """تحويل آمن لأي قيمة إلى int"""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return int(f)
    except (ValueError, TypeError):
        return None


def read_csv_smart(content_bytes):
    """قراءة CSV مع محاولة encodings مختلفة"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for enc in encodings:
        try:
            content = content_bytes.decode(enc)
            # محاولة separators مختلفة
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep, low_memory=False)
                    if df.shape[1] > 1:  # لو في أكتر من عمود يبقى الـ separator صحيح
                        return df
                except Exception:
                    continue
            # fallback: comma default
            return pd.read_csv(io.StringIO(content), low_memory=False)
        except UnicodeDecodeError:
            continue
    
    raise ValueError("Could not decode CSV file with any known encoding")


def compute_histogram_bins(series, num_bins=15):
    """حساب bins للـ histogram جاهزة للرسم في الموبايل"""
    try:
        clean = series.dropna()
        if len(clean) == 0:
            return []
        
        counts, edges = np.histogram(clean, bins=num_bins)
        bins = []
        for i in range(len(counts)):
            bins.append({
                "bin_start": safe_float(edges[i]),
                "bin_end": safe_float(edges[i + 1]),
                "count": int(counts[i]),
                "label": f"{safe_float(edges[i])}-{safe_float(edges[i+1])}"
            })
        return bins
    except Exception:
        return []


def compute_column_summary(numeric_df, col):
    """حساب الإحصائيات الكاملة لعمود رقمي واحد"""
    s = numeric_df[col].dropna()
    
    # لو العمود فاضي
    if len(s) == 0:
        return {
            "mean": None, "median": None, "mode": "N/A",
            "std": None, "variance": None, "min": None, "max": None,
            "q1": None, "q3": None, "iqr": None,
            "skewness": None, "kurtosis": None,
            "outlier_count": 0, "outlier_pct": 0.0,
            "null_count": int(numeric_df[col].isnull().sum()),
            "count": 0,
            "scatter_data": [],
            "outlier_points": [],
            "sample_values": [],
            "histogram_bins": []
        }

    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers_mask = (s < lower) | (s > upper)
    outlier_count = int(outliers_mask.sum())

    # ═══ Scatter data (500 sample points بدل 200) ═══
    sample_size = min(500, len(s))
    sample = s.sample(sample_size, random_state=42).reset_index(drop=True)
    scatter_data = [
        {"x": round(float(i), 4), "y": round(float(v), 4)}
        for i, v in enumerate(sample)
        if not math.isnan(float(v))
    ]

    # ═══ Outlier points (100 نقطة بدل 50) ═══
    outlier_vals = s[outliers_mask].head(100).reset_index(drop=True)
    outlier_points = [
        {"x": round(float(i), 4), "y": round(float(v), 4)}
        for i, v in enumerate(outlier_vals)
        if not math.isnan(float(v))
    ]

    # ═══ Sample values (100 قيمة عشوائية للرسم في الموبايل) ═══
    sample_for_display = min(100, len(s))
    sample_vals = s.sample(sample_for_display, random_state=42).tolist()
    sample_values = [safe_float(v) for v in sample_vals if safe_float(v) is not None]

    # ═══ Histogram bins (15 bin جاهزة) ═══
    histogram_bins = compute_histogram_bins(s, num_bins=15)

    # ═══ Mode ═══
    try:
        mode_val = str(round(float(s.mode()[0]), 4)) if not s.mode().empty else "N/A"
    except Exception:
        mode_val = "N/A"

    return {
        "mean":           safe_float(s.mean()),
        "median":         safe_float(s.median()),
        "mode":           mode_val,
        "std":            safe_float(s.std()),
        "variance":       safe_float(s.var()),
        "min":            safe_float(s.min()),
        "max":            safe_float(s.max()),
        "q1":             safe_float(q1),
        "q3":             safe_float(q3),
        "iqr":            safe_float(iqr),
        "skewness":       safe_float(s.skew()),
        "kurtosis":       safe_float(s.kurtosis()),
        "outlier_count":  outlier_count,
        "outlier_pct":    safe_float(outlier_count / len(s) * 100) or 0.0,
        "null_count":     int(numeric_df[col].isnull().sum()),
        "count":          int(s.count()),
        "scatter_data":   scatter_data,
        "outlier_points": outlier_points,
        "sample_values":  sample_values,
        "histogram_bins": histogram_bins
    }


# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/')
def home():
    return jsonify({
        "message": "GoBike Analysis API is running! 🚲",
        "version": "2.0",
        "endpoints": {
            "/stats": "GET - GoBike demo statistics",
            "/analyze": "POST - Upload and analyze CSV",
            "/clean": "POST - Clean CSV data",
            "/health": "GET - Server health check"
        }
    })


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__
    })


@app.route('/stats')
def stats():
    """GoBike Demo Stats مع scatter_pairs حقيقي و histogram_bins"""
    return jsonify({
        "total_rows":           183416,
        "total_columns":        16,
        "duplicates":           0,
        "missing_pct":          0.0,
        "total_missing_values": 0,
        "columns":              ["duration_min", "age", "user_type", "member_gender"],
        "numeric_cols":         ["duration_min", "age"],
        "numeric_summary": {
            "duration_min": {
                "mean":           12.1,
                "median":         8.5,
                "mode":           "5.0",
                "std":            10.2,
                "variance":       104.04,
                "min":            1.0,
                "max":            1440.0,
                "q1":             5.2,
                "q3":             16.3,
                "iqr":            11.1,
                "skewness":       3.2,
                "kurtosis":       15.4,
                "outlier_count":  1200,
                "outlier_pct":    0.65,
                "null_count":     0,
                "count":          183416,
                "scatter_data":   [],
                "outlier_points": [],
                "sample_values":  [5.2, 8.1, 12.5, 15.3, 20.1, 25.6, 30.2, 35.8, 42.1, 50.5],
                "histogram_bins": [
                    {"bin_start": 1.0,   "bin_end": 100.0,  "count": 165000, "label": "1-100"},
                    {"bin_start": 100.0, "bin_end": 300.0,  "count": 15000,  "label": "100-300"},
                    {"bin_start": 300.0, "bin_end": 600.0,  "count": 2500,   "label": "300-600"},
                    {"bin_start": 600.0, "bin_end": 1440.0, "count": 916,    "label": "600-1440"}
                ]
            },
            "age": {
                "mean":           40.2,
                "median":         35.0,
                "mode":           "28.0",
                "std":            10.5,
                "variance":       110.25,
                "min":            18.0,
                "max":            120.0,
                "q1":             28.0,
                "q3":             40.0,
                "iqr":            12.0,
                "skewness":       1.8,
                "kurtosis":       5.2,
                "outlier_count":  500,
                "outlier_pct":    0.27,
                "null_count":     0,
                "count":          183416,
                "scatter_data":   [],
                "outlier_points": [],
                "sample_values":  [25, 30, 35, 28, 45, 50, 33, 40, 55, 60],
                "histogram_bins": [
                    {"bin_start": 18.0, "bin_end": 30.0,  "count": 55000, "label": "18-30"},
                    {"bin_start": 30.0, "bin_end": 45.0,  "count": 90000, "label": "30-45"},
                    {"bin_start": 45.0, "bin_end": 60.0,  "count": 35000, "label": "45-60"},
                    {"bin_start": 60.0, "bin_end": 120.0, "count": 3416,  "label": "60-120"}
                ]
            }
        },
        "categorical_cols": ["user_type", "member_gender"],
        "cat_summary": {
            "user_type": {
                "unique_values": 2,
                "null_count":    0,
                "top_values": {
                    "Subscriber": 163567,
                    "Customer":   19849
                }
            },
            "member_gender": {
                "unique_values": 3,
                "null_count":    0,
                "top_values": {
                    "Male":   138838,
                    "Female": 40351,
                    "Other":  4227
                }
            }
        },
        "correlation": {
            "duration_min": {"duration_min": 1.0,  "age": -0.05},
            "age":          {"duration_min": -0.05, "age": 1.0}
        },
        "scatter_pairs": [
            {
                "x_col": "duration_min",
                "y_col": "age",
                "points": [
                    {"x": 5.2, "y": 25}, {"x": 8.1, "y": 30}, {"x": 12.5, "y": 35},
                    {"x": 15.3, "y": 28}, {"x": 20.1, "y": 45}, {"x": 25.6, "y": 50},
                    {"x": 30.2, "y": 33}, {"x": 35.8, "y": 40}, {"x": 42.1, "y": 55},
                    {"x": 50.5, "y": 60}
                ]
            }
        ]
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # ═══ قراءة الملف بذكاء ═══
        content_bytes = file.read()
        try:
            df = read_csv_smart(content_bytes)
        except Exception as e:
            return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

        # ─── Basic Info ───
        total_rows    = int(df.shape[0])
        total_columns = int(df.shape[1])
        columns       = list(df.columns)
        duplicates    = int(df.duplicated().sum())

        # ─── Data Types Info ───
        data_types = {}
        for col in columns:
            dtype = str(df[col].dtype)
            if 'int' in dtype or 'float' in dtype:
                data_types[col] = "numeric"
            elif 'bool' in dtype:
                data_types[col] = "boolean"
            elif 'datetime' in dtype:
                data_types[col] = "datetime"
            else:
                data_types[col] = "categorical"

        # ─── Missing Values ───
        missing_dict = {
            col: int(val)
            for col, val in df.isnull().sum().items()
            if val > 0
        }
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = safe_float(
            df.isnull().sum().sum() / total_cells * 100
        ) if total_cells > 0 else 0.0
        missing_pct = missing_pct or 0.0
        total_missing = int(df.isnull().sum().sum())

        # ─── Numeric Summary ───
        numeric_df   = df.select_dtypes(include='number')
        numeric_cols = list(numeric_df.columns)
        numeric_summary = {}

        for col in numeric_cols:
            numeric_summary[col] = compute_column_summary(numeric_df, col)

        # ─── Categorical Summary ───
        cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        cat_summary = {}

        for col in cat_cols:
            try:
                top_values   = df[col].value_counts().head(15)  # زودنا لـ 15 بدل 10
                unique_count = int(df[col].nunique())
                null_count   = int(df[col].isnull().sum())

                cat_summary[col] = {
                    "unique_values": unique_count,
                    "null_count":    null_count,
                    "top_values": {
                        str(k): int(v)
                        for k, v in top_values.items()
                    }
                }
            except Exception as e:
                cat_summary[col] = {
                    "unique_values": 0,
                    "null_count": int(df[col].isnull().sum()),
                    "top_values": {}
                }

        # ─── Correlation Matrix (كل الأعمدة الرقمية) ───
        correlation = {}
        if len(numeric_cols) > 1:
            try:
                corr_matrix = numeric_df.corr()
                correlation = {
                    col: {
                        col2: safe_float(val) or 0.0
                        for col2, val in row.items()
                    }
                    for col, row in corr_matrix.to_dict().items()
                }
            except Exception:
                correlation = {}

        # ─── Scatter Pairs (كل الأعمدة الرقمية بحد أقصى 10) ───
        scatter_pairs = []
        if len(numeric_cols) >= 2:
            # حد أقصى 10 أعمدة رقمية عشان الأداء
            cols_for_pairs = numeric_cols[:10]
            max_pairs = 20  # حد أقصى 20 pair لتوازن الحجم والأداء
            pair_count = 0
            
            for i in range(len(cols_for_pairs)):
                if pair_count >= max_pairs:
                    break
                for j in range(i + 1, len(cols_for_pairs)):
                    if pair_count >= max_pairs:
                        break
                        
                    col_x = cols_for_pairs[i]
                    col_y = cols_for_pairs[j]
                    
                    try:
                        paired = df[[col_x, col_y]].dropna()
                        if len(paired) == 0:
                            continue

                        # 300 نقطة لكل pair
                        sample_size = min(300, len(paired))
                        sample = paired.sample(sample_size, random_state=42)
                        
                        points = []
                        for _, row in sample.iterrows():
                            x = safe_float(row[col_x])
                            y = safe_float(row[col_y])
                            if x is not None and y is not None:
                                points.append({"x": x, "y": y})

                        if points:  # ماتضيفش pair فاضي
                            scatter_pairs.append({
                                "x_col":  col_x,
                                "y_col":  col_y,
                                "points": points
                            })
                            pair_count += 1
                    except Exception:
                        continue

        # ─── الاستجابة النهائية ───
        return jsonify({
            "total_rows":           total_rows,
            "total_columns":        total_columns,
            "columns":              columns,
            "data_types":           data_types,
            "duplicates":           duplicates,
            "missing_values":       missing_dict,
            "missing_pct":          missing_pct,
            "total_missing_values": total_missing,
            "numeric_cols":         numeric_cols,
            "numeric_summary":      numeric_summary,
            "categorical_cols":     cat_cols,
            "cat_summary":          cat_summary,
            "correlation":          correlation,
            "scatter_pairs":        scatter_pairs
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route('/clean', methods=['POST'])
def clean():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        content_bytes = file.read()
        
        try:
            df = read_csv_smart(content_bytes)
        except Exception as e:
            return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

        before = {
            "rows":       int(df.shape[0]),
            "missing":    int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum())
        }

        # ─── إزالة المكرر ───
        df = df.drop_duplicates()

        # ─── ملء القيم الفارغة (بدون inplace warning) ───
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            mean_val = df[col].mean()
            if pd.notna(mean_val):
                df[col] = df[col].fillna(mean_val)

        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            mode_series = df[col].mode()
            if not mode_series.empty:
                df[col] = df[col].fillna(mode_series[0])

        after = {
            "rows":       int(df.shape[0]),
            "missing":    int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum())
        }

        # نسبة التحسن
        removed_rows = before["rows"] - after["rows"]
        filled_missing = before["missing"] - after["missing"]

        return jsonify({
            "before": before,
            "after":  after,
            "changes": {
                "duplicates_removed": removed_rows,
                "missing_filled":     filled_missing
            },
            "message": "Data cleaned successfully! ✅"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Cleaning failed: {str(e)}"}), 500


# ═══════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "File too large. Maximum size allowed is 50MB"}), 413


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

# حد أقصى لحجم الملف: 50 ميجا
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
