from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import math

app = Flask(__name__)
CORS(app)


def safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 4)
    except:
        return None


@app.route('/')
def home():
    return "GoBike API is running! 🚲"


@app.route('/stats')
def stats():
    return jsonify({
        "total_rows":           183416,
        "total_columns":        16,
        "duplicates":           0,
        "missing_pct":          0.0,
        "total_missing_values": 0,
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
                "outlier_points": []
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
                "outlier_points": []
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
        "correlation":   {},
        "scatter_pairs": []
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        content = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))

        # ─── Basic Info ───
        total_rows    = int(df.shape[0])
        total_columns = int(df.shape[1])
        columns       = list(df.columns)
        duplicates    = int(df.duplicated().sum())

        missing_dict = {
            col: int(val)
            for col, val in df.isnull().sum().items()
            if val > 0
        }
        missing_pct   = safe_float(
            df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        ) or 0.0
        total_missing = int(df.isnull().sum().sum())

        # ─── Numeric Summary ───
        numeric_df   = df.select_dtypes(include='number')
        numeric_cols = list(numeric_df.columns)
        numeric_summary = {}

        for col in numeric_cols:
            s   = numeric_df[col].dropna()

            if len(s) == 0:
                numeric_summary[col] = {
                    "mean": None, "median": None, "mode": "N/A",
                    "std": None, "variance": None, "min": None, "max": None,
                    "q1": None, "q3": None, "iqr": None,
                    "skewness": None, "kurtosis": None,
                    "outlier_count": 0, "outlier_pct": 0.0,
                    "null_count": int(numeric_df[col].isnull().sum()),
                    "count": 0, "scatter_data": [], "outlier_points": []
                }
                continue

            q1  = float(s.quantile(0.25))
            q3  = float(s.quantile(0.75))
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers_mask = (s < lower) | (s > upper)
            outlier_count = int(outliers_mask.sum())

            # Scatter data
            sample_size  = min(200, len(s))
            sample       = s.sample(sample_size, random_state=42).reset_index(drop=True)
            scatter_data = [
                {"x": round(float(i), 4), "y": round(float(v), 4)}
                for i, v in enumerate(sample)
                if not math.isnan(float(v))
            ]

            # Outlier points
            outlier_vals   = s[outliers_mask].head(50).reset_index(drop=True)
            outlier_points = [
                {"x": round(float(i), 4), "y": round(float(v), 4)}
                for i, v in enumerate(outlier_vals)
                if not math.isnan(float(v))
            ]

            try:
                mode_val = str(round(float(s.mode()[0]), 4)) if not s.mode().empty else "N/A"
            except Exception:
                mode_val = "N/A"

            numeric_summary[col] = {
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
                "outlier_points": outlier_points
            }

        # ─── Categorical Summary ───
        cat_cols = list(df.select_dtypes(include='object').columns)
        cat_summary = {}

        for col in cat_cols:
            top_values   = df[col].value_counts().head(10)
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

        # ─── Correlation ───
        correlation = {}
        if len(numeric_cols) > 1:
            corr_matrix = numeric_df.corr()
            # عالج الـ NaN في الـ correlation
            correlation = {
                col: {
                    col2: safe_float(val) or 0.0
                    for col2, val in row.items()
                }
                for col, row in corr_matrix.to_dict().items()
            }

        # ─── Scatter Pairs ───
        scatter_pairs = []
        if len(numeric_cols) >= 2:
            for i in range(min(3, len(numeric_cols))):
                for j in range(i + 1, min(4, len(numeric_cols))):
                    col_x  = numeric_cols[i]
                    col_y  = numeric_cols[j]
                    paired = df[[col_x, col_y]].dropna()

                    if len(paired) == 0:
                        continue

                    sample = paired.sample(
                        min(200, len(paired)), random_state=42
                    )
                    points = []
                    for _, row in sample.iterrows():
                        x = safe_float(row[col_x])
                        y = safe_float(row[col_y])
                        if x is not None and y is not None:
                            points.append({"x": x, "y": y})

                    scatter_pairs.append({
                        "x_col":  col_x,
                        "y_col":  col_y,
                        "points": points
                    })

        return jsonify({
            "total_rows":           total_rows,
            "total_columns":        total_columns,
            "columns":              columns,
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
        return jsonify({"error": str(e)}), 500


@app.route('/clean', methods=['POST'])
def clean():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file    = request.files['file']
        content = file.read().decode('utf-8')
        df      = pd.read_csv(io.StringIO(content))

        before = {
            "rows":       int(df.shape[0]),
            "missing":    int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum())
        }

        df = df.drop_duplicates()

        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)

        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)

        after = {
            "rows":       int(df.shape[0]),
            "missing":    int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum())
        }

        return jsonify({
            "before":  before,
            "after":   after,
            "message": "Data cleaned successfully!"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
