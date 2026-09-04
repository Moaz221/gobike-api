from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import pandas as pd
import numpy as np
import io
import time
import logging
from logging.handlers import RotatingFileHandler
import traceback
import os
import json
import base64
import math
import hashlib
import chardet
from typing import Dict, Any, Optional, List

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

class Config:
    API_VERSION = "2.1.0"
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
    RATE_LIMIT = os.getenv('RATE_LIMIT', '10 per minute')
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', 300))
    MAX_SCATTER_POINTS = 100
    MAX_SAMPLE_SIZE = 120
    MAX_CORRELATION_COLS = 12
    MAX_HISTOGRAM_BINS = 12
    MAX_CATEGORICAL_VALUES = 10
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ═══════════════════════════════════════════════════════════
# APP INITIALIZATION
# ═══════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# CORS
CORS(app, resources={
    r"/*": {
        "origins": Config.ALLOWED_ORIGINS,
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Rate Limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[Config.RATE_LIMIT],
    storage_uri="memory://"
)

# Caching
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': Config.CACHE_DEFAULT_TIMEOUT,
    'CACHE_THRESHOLD': 100
})

# Logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def safe_float(val):
    """تحويل آمن لقيمة float"""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 4)
    except (ValueError, TypeError, OverflowError):
        return None

def safe_int(val):
    """تحويل آمن لقيمة int"""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return int(f)
    except (ValueError, TypeError, OverflowError):
        return None

def get_file_hash(content):
    """إنشاء hash فريد للملف"""
    return hashlib.md5(content).hexdigest()

def detect_encoding(content_bytes):
    """اكتشاف ترميز الملف"""
    try:
        result = chardet.detect(content_bytes[:10000])
        confidence = result.get('confidence', 0)
        encoding = result.get('encoding', 'utf-8')
        
        if confidence < 0.7:
            for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    content_bytes.decode(enc)
                    return enc
                except UnicodeDecodeError:
                    continue
        
        return encoding if encoding else 'utf-8'
    except:
        return 'utf-8'

def detect_separator(sample_text):
    """اكتشاف الفاصل المستخدم"""
    separators = [',', ';', '\t', '|']
    best_sep = ','
    best_count = 0
    
    first_line = sample_text.split('\n')[0] if sample_text else ''
    
    for sep in separators:
        count = first_line.count(sep)
        if count > best_count:
            best_count = count
            best_sep = sep
    
    return best_sep

def read_csv_smart(content_bytes):
    """قراءة CSV بذكاء"""
    try:
        encoding = detect_encoding(content_bytes)
        content = content_bytes.decode(encoding, errors='ignore')
        
        sample = content[:2000]
        sep = detect_separator(sample)
        
        try:
            df = pd.read_csv(io.StringIO(content), sep=sep, low_memory=False, engine='python')
            if df.shape[1] > 1:
                return df
            df = pd.read_csv(io.StringIO(content), low_memory=False)
            return df
        except Exception:
            df = pd.read_csv(io.StringIO(content), low_memory=False)
            return df
            
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise ValueError(f"Could not decode CSV file: {str(e)}")

def compute_histogram_bins(series, num_bins=12):
    """حساب bins للرسم البياني"""
    try:
        clean = series.dropna()
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean) == 0:
            return []
        
        optimal_bins = min(num_bins, max(5, int(np.sqrt(len(clean)))))
        counts, edges = np.histogram(clean, bins=optimal_bins)
        
        bins = []
        for i in range(len(counts)):
            start = safe_float(edges[i])
            end = safe_float(edges[i + 1])
            bins.append({
                "bin_start": start,
                "bin_end": end,
                "count": int(counts[i]),
                "label": f"{start}-{end}"
            })
        return bins
    except Exception as e:
        logger.warning(f"Error computing histogram: {e}")
        return []

def compute_mode(s):
    """حساب mode بأمان"""
    try:
        mode_raw = s.mode()
        if len(mode_raw) > 0:
            val = mode_raw.iloc[0]
            if isinstance(val, (int, float)):
                return str(round(float(val), 4))
            return str(val)
        return "N/A"
    except:
        return "N/A"

def compute_scatter_data(s, max_points=80):
    """إنشاء scatter data"""
    try:
        n_points = min(max_points, len(s))
        if n_points == 0:
            return []
        
        indices = np.linspace(0, len(s) - 1, n_points, dtype=int)
        sampled = s.iloc[indices]
        
        return [
            {"x": float(i), "y": safe_float(val)}
            for i, val in enumerate(sampled)
            if safe_float(val) is not None
        ]
    except:
        return []

def compute_outlier_points(s, outliers_mask, max_points=40):
    """حساب نقاط outliers"""
    try:
        outlier_vals = s[outliers_mask]
        if len(outlier_vals) == 0:
            return []
        
        if len(outlier_vals) > max_points:
            outlier_vals = outlier_vals.sample(max_points, random_state=42)
        
        return [
            {"x": float(i), "y": safe_float(val)}
            for i, val in enumerate(outlier_vals)
            if safe_float(val) is not None
        ]
    except:
        return []

def compute_sample_values(s, max_sample=120):
    """حساب قيم عينة"""
    try:
        n_sample = min(max_sample, len(s))
        sample_vals = s.sample(n_sample, random_state=42).astype(float).tolist()
        return [v for v in map(safe_float, sample_vals) if v is not None]
    except:
        return []

def compute_column_summary(series_full, col_name, include_visualizations=True):
    """حساب إحصائيات عمود"""
    null_count = int(series_full.isnull().sum())
    s = series_full.dropna()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(s) == 0:
        return {
            "mean": None, "median": None, "mode": "N/A",
            "std": None, "variance": None, "min": None, "max": None,
            "q1": None, "q3": None, "iqr": None,
            "skewness": None, "kurtosis": None,
            "outlier_count": 0, "outlier_pct": 0.0,
            "null_count": null_count,
            "count": 0,
            "scatter_data": [], "outlier_points": [],
            "sample_values": [], "histogram_bins": []
        }
    
    try:
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        outliers_mask = (s < lower) | (s > upper)
        outlier_count = int(outliers_mask.sum())
        outlier_pct = (outlier_count / len(s)) * 100.0 if outlier_count > 0 else 0.0
        
        result = {
            "mean": safe_float(s.mean()),
            "median": safe_float(s.median()),
            "mode": compute_mode(s),
            "std": safe_float(s.std()),
            "variance": safe_float(s.var()),
            "min": safe_float(s.min()),
            "max": safe_float(s.max()),
            "q1": safe_float(q1),
            "q3": safe_float(q3),
            "iqr": safe_float(iqr),
            "skewness": safe_float(s.skew()),
            "kurtosis": safe_float(s.kurtosis()),
            "outlier_count": outlier_count,
            "outlier_pct": safe_float(outlier_pct),
            "null_count": null_count,
            "count": int(s.count()),
        }
        
        if include_visualizations:
            result.update({
                "scatter_data": compute_scatter_data(s),
                "outlier_points": compute_outlier_points(s, outliers_mask),
                "sample_values": compute_sample_values(s),
                "histogram_bins": compute_histogram_bins(s)
            })
        else:
            result.update({
                "scatter_data": [],
                "outlier_points": [],
                "sample_values": [],
                "histogram_bins": []
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error computing summary for {col_name}: {e}")
        return {
            "mean": None, "median": None, "mode": "N/A",
            "std": None, "variance": None, "min": None, "max": None,
            "q1": None, "q3": None, "iqr": None,
            "skewness": None, "kurtosis": None,
            "outlier_count": 0, "outlier_pct": 0.0,
            "null_count": null_count,
            "count": int(s.count()),
            "scatter_data": [], "outlier_points": [],
            "sample_values": [], "histogram_bins": []
        }

def build_scatter_pairs(df, numeric_cols, max_cols=8, max_pairs=15, max_points=100):
    """بناء scatter pairs"""
    scatter_pairs = []
    if len(numeric_cols) < 2:
        return scatter_pairs
    
    cols = numeric_cols[:max_cols]
    pair_count = 0
    
    for i in range(len(cols)):
        if pair_count >= max_pairs:
            break
        for j in range(i + 1, len(cols)):
            if pair_count >= max_pairs:
                break
            
            col_x = cols[i]
            col_y = cols[j]
            
            try:
                paired = df[[col_x, col_y]].dropna()
                if len(paired) < 2:
                    continue
                
                if len(paired) > max_points:
                    paired = paired.sample(max_points, random_state=42)
                
                xs = paired[col_x].astype(float).values
                ys = paired[col_y].astype(float).values
                
                points = []
                for x, y in zip(xs, ys):
                    sx = safe_float(x)
                    sy = safe_float(y)
                    if sx is not None and sy is not None:
                        points.append({"x": sx, "y": sy})
                
                if len(points) >= 2:
                    scatter_pairs.append({
                        "x_col": col_x,
                        "y_col": col_y,
                        "points": points
                    })
                    pair_count += 1
            except Exception as e:
                logger.warning(f"Error building scatter pair {col_x}-{col_y}: {e}")
                continue
    
    return scatter_pairs

def get_data_types(df):
    """تحديد أنواع البيانات"""
    data_types = {}
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                data_types[col] = "numeric"
            elif pd.api.types.is_bool_dtype(df[col]):
                data_types[col] = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                data_types[col] = "datetime"
            else:
                data_types[col] = "categorical"
        except:
            data_types[col] = "categorical"
    return data_types

def compute_correlation(numeric_df, numeric_cols):
    """حساب correlation"""
    correlation = {}
    if len(numeric_cols) <= 1:
        return correlation
    
    try:
        corr_cols = numeric_cols[:Config.MAX_CORRELATION_COLS]
        valid_cols = [col for col in corr_cols if numeric_df[col].std() > 0]
        
        if len(valid_cols) > 1:
            corr_matrix = numeric_df[valid_cols].corr()
            correlation = {
                col: {
                    col2: safe_float(val) or 0.0
                    for col2, val in row.items()
                }
                for col, row in corr_matrix.to_dict().items()
            }
    except Exception as e:
        logger.warning(f"Error computing correlation: {e}")
    
    return correlation

def analyze_dataframe(df, include_visualizations=True):
    """تحليل DataFrame كامل"""
    df.columns = [str(c).strip() for c in df.columns]
    
    total_rows = int(df.shape[0])
    total_columns = int(df.shape[1])
    columns = list(df.columns)
    duplicates = int(df.duplicated().sum())
    
    data_types = get_data_types(df)
    
    missing_dict = {
        col: int(val)
        for col, val in df.isnull().sum().items()
        if val > 0
    }
    total_cells = max(total_rows * total_columns, 1)
    total_missing = int(df.isnull().sum().sum())
    missing_pct = (total_missing / total_cells) * 100.0 if total_missing > 0 else 0.0
    
    numeric_df = df.select_dtypes(include='number')
    numeric_cols = list(numeric_df.columns)
    
    numeric_summary = {}
    for col in numeric_cols:
        numeric_summary[col] = compute_column_summary(
            numeric_df[col], col, include_visualizations
        )
    
    cat_cols = list(df.select_dtypes(include=['object', 'category', 'bool']).columns)
    cat_summary = {}
    for col in cat_cols:
        try:
            top_values = df[col].value_counts(dropna=True).head(Config.MAX_CATEGORICAL_VALUES)
            cat_summary[col] = {
                "unique_values": int(df[col].nunique(dropna=True)),
                "null_count": int(df[col].isnull().sum()),
                "top_values": {str(k): int(v) for k, v in top_values.items()}
            }
        except Exception as e:
            logger.warning(f"Error computing categorical summary for {col}: {e}")
            cat_summary[col] = {
                "unique_values": 0,
                "null_count": int(df[col].isnull().sum()),
                "top_values": {}
            }
    
    correlation = compute_correlation(numeric_df, numeric_cols)
    
    scatter_pairs = build_scatter_pairs(
        df=df,
        numeric_cols=numeric_cols,
        max_cols=8,
        max_pairs=15,
        max_points=Config.MAX_SCATTER_POINTS
    )
    
    return {
        "api_version": Config.API_VERSION,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "columns": columns,
        "data_types": data_types,
        "duplicates": duplicates,
        "missing_values": missing_dict,
        "missing_pct": safe_float(missing_pct),
        "total_missing_values": total_missing,
        "numeric_cols": numeric_cols,
        "numeric_summary": numeric_summary,
        "categorical_cols": cat_cols,
        "cat_summary": cat_summary,
        "correlation": correlation,
        "scatter_pairs": scatter_pairs
    }

def clean_dataframe(df):
    """تنظيف DataFrame"""
    before = {
        "rows": int(df.shape[0]),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum())
    }
    
    df = df.drop_duplicates()
    
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        median_val = df[col].median()
        if pd.notna(median_val):
            df[col] = df[col].fillna(median_val)
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        mode_series = df[col].mode()
        if not mode_series.empty:
            df[col] = df[col].fillna(mode_series.iloc[0])
    
    after = {
        "rows": int(df.shape[0]),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum())
    }
    
    return df, before, after

def load_demo_stats():
    """تحميل الإحصائيات التجريبية"""
    try:
        with open('data/stats.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {
            "total_trips": 183416,
            "avg_duration_min": 12.1,
            "avg_age": 40.2,
            "top_user_type": "Subscriber",
            "top_gender": "Male",
            "subscriber_pct": 89.2,
            "male_pct": 75.7
        }

# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/')
def home():
    """Home page"""
    return jsonify({
        "message": "GoBike Analysis API",
        "version": Config.API_VERSION,
        "status": "running",
        "endpoints": {
            "/stats": "GET - GoBike demo statistics",
            "/analyze": "POST - Upload and analyze CSV file",
            "/clean": "POST - Clean CSV data",
            "/health": "GET - Server health check"
        }
    })

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "api_version": Config.API_VERSION,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "timestamp": int(time.time())
    })

@app.route('/stats')
@cache.cached(timeout=Config.CACHE_DEFAULT_TIMEOUT)
def stats():
    """Get demo statistics"""
    try:
        demo_stats = load_demo_stats()
        
        response = {
            "api_version": Config.API_VERSION,
            "total_trips": demo_stats.get("total_trips", 183416),
            "summary": {
                "avg_duration_min": demo_stats.get("avg_duration_min", 12.1),
                "avg_age": demo_stats.get("avg_age", 40.2)
            },
            "top_categories": {
                "user_type": demo_stats.get("top_user_type", "Subscriber"),
                "gender": demo_stats.get("top_gender", "Male")
            },
            "percentages": {
                "subscriber_pct": demo_stats.get("subscriber_pct", 89.2),
                "male_pct": demo_stats.get("male_pct", 75.7)
            }
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error loading stats: {e}")
        return jsonify({"error": "Failed to load statistics"}), 500

@app.route('/analyze', methods=['POST'])
@limiter.limit(Config.RATE_LIMIT)
def analyze():
    """Analyze uploaded CSV file"""
    t0 = time.time()
    
    try:
        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "message": "Please upload a CSV file using the 'file' field"
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "message": "Please select a file to upload"
            }), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                "error": "Invalid file type",
                "message": "File must be a CSV file"
            }), 400
        
        content_bytes = file.read()
        
        file_hash = get_file_hash(content_bytes)
        cache_key = f"analysis_{file_hash}"
        
        cached_result = cache.get(cache_key)
        if cached_result:
            cached_result['cached'] = True
            cached_result['message'] = "Result retrieved from cache"
            return jsonify(cached_result)
        
        try:
            df = read_csv_smart(content_bytes)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return jsonify({
                "error": "Failed to read CSV",
                "message": "The file could not be read. Please check the file format."
            }), 400
        
        include_viz = request.args.get('include_viz', 'true').lower() == 'true'
        
        result = analyze_dataframe(df, include_viz)
        
        elapsed_ms = int((time.time() - t0) * 1000)
        result['processing_ms'] = elapsed_ms
        result['cached'] = False
        result['message'] = "File analyzed successfully"
        
        cache.set(cache_key, result, timeout=Config.CACHE_DEFAULT_TIMEOUT)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Analysis failed",
            "message": "An error occurred while analyzing the file. Please try again."
        }), 500

@app.route('/clean', methods=['POST'])
@limiter.limit(Config.RATE_LIMIT)
def clean():
    """Clean uploaded CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "message": "Please upload a CSV file using the 'file' field"
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "message": "Please select a file to upload"
            }), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                "error": "Invalid file type",
                "message": "File must be a CSV file"
            }), 400
        
        content_bytes = file.read()
        
        try:
            df = read_csv_smart(content_bytes)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return jsonify({
                "error": "Failed to read CSV",
                "message": "The file could not be read. Please check the file format."
            }), 400
        
        df_cleaned, before, after = clean_dataframe(df)
        
        csv_buffer = io.StringIO()
        df_cleaned.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_string = csv_buffer.getvalue()
        
        csv_base64 = base64.b64encode(csv_string.encode('utf-8')).decode('utf-8')
        
        original_name = file.filename.rsplit('.', 1)[0]
        clean_filename = f"cleaned_{original_name}.csv"
        
        return jsonify({
            "success": True,
            "message": "File cleaned successfully",
            "cleaning_stats": {
                "before": before,
                "after": after,
                "duplicates_removed": before['duplicates'],
                "missing_values_filled": before['missing'] - after['missing'],
                "rows_removed": before['rows'] - after['rows']
            },
            "cleaned_file": {
                "filename": clean_filename,
                "content_base64": csv_base64,
                "size_bytes": len(csv_string.encode('utf-8')),
                "encoding": "utf-8",
                "format": "csv"
            }
        })
        
    except Exception as e:
        logger.error(f"Cleaning failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Cleaning failed",
            "message": "An error occurred while cleaning the file. Please try again."
        }), 500

# ═══════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "message": "The request could not be understood"}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "message": "The requested endpoint does not exist"}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large", "message": f"Maximum file size is {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB"}), 413

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded", "message": "Too many requests. Please try again later."}), 429

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error", "message": "An unexpected error occurred"}), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    logger.error(f"Unhandled exception: {e}")
    logger.error(traceback.format_exc())
    return jsonify({"error": "Unexpected error", "message": "An unexpected error occurred. Please try again later."}), 500

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(
        debug=False,
        host='0.0.0.0',
        port=port,
        threaded=True
    )