from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import pandas as pd
import io

app = Flask(__name__)
CORS(app)

try:
    with open('results.json') as f:
        gobike_data = json.load(f)
except FileNotFoundError:
    gobike_data = {"error": "results.json not found"}


@app.route('/')
def home():
    return "GoBike API is running! 🚲"


@app.route('/stats')
def stats():
    return jsonify(gobike_data)


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

        total_rows = int(df.shape[0])
        total_columns = int(df.shape[1])
        columns = list(df.columns)
        duplicates = int(df.duplicated().sum())

        missing_dict = {
            col: int(val)
            for col, val in df.isnull().sum().items()
            if val > 0
        }
        missing_pct = round(
            float(df.isnull().sum().sum() /
            (df.shape[0] * df.shape[1]) * 100), 2
        )

        numeric_df = df.select_dtypes(include='number')
        numeric_cols = list(numeric_df.columns)

        numeric_summary = {}
        for col in numeric_cols:
            numeric_summary[col] = {
                "mean": round(float(numeric_df[col].mean()), 2),
                "min":  round(float(numeric_df[col].min()), 2),
                "max":  round(float(numeric_df[col].max()), 2),
                "std":  round(float(numeric_df[col].std()), 2)
            }

        cat_cols = list(df.select_dtypes(include='object').columns)
        cat_summary = {}
        for col in cat_cols:
            top_values = df[col].value_counts().head(5)
            cat_summary[col] = {
                str(k): int(v)
                for k, v in top_values.items()
            }

        return jsonify({
            "total_rows":       total_rows,
            "total_columns":    total_columns,
            "columns":          columns,
            "duplicates":       duplicates,
            "missing_values":   missing_dict,
            "missing_pct":      missing_pct,
            "numeric_cols":     numeric_cols,
            "numeric_summary":  numeric_summary,
            "categorical_cols": cat_cols,
            "cat_summary":      cat_summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)