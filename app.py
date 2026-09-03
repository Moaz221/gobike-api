from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import pandas as pd
import io

app = Flask(__name__)
CORS(app)

# تحميل نتائج GoBike الجاهزة
with open('results.json') as f:
    gobike_data = json.load(f)


# ─── Route 1: تأكد إن الـ API شغال ───
@app.route('/')
def home():
    return "GoBike API is running! 🚲"


# ─── Route 2: نتائج GoBike الجاهزة ───
@app.route('/stats')
def stats():
    return jsonify(gobike_data)


# ─── Route 3: تحليل أي CSV جديد ───
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # استقبال الملف
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # قراءة الـ CSV
        content = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))

        # ─── التحليل ───
        # معلومات أساسية
        total_rows = int(df.shape[0])
        total_columns = int(df.shape[1])
        columns = list(df.columns)
        duplicates = int(df.duplicated().sum())

        # Missing Values
        missing = df.isnull().sum()
        missing_dict = {
            col: int(val)
            for col, val in missing.items()
            if val > 0
        }
        missing_pct = round(
            float(df.isnull().sum().sum() /
            (df.shape[0] * df.shape[1]) * 100), 2
        )

        # الـ Numeric columns
        numeric_df = df.select_dtypes(include='number')
        numeric_cols = list(numeric_df.columns)

        # إحصائيات الـ Numeric columns
        numeric_summary = {}