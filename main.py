from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from flask_compress import Compress
import pandas as pd
import numpy as np
import io
import time
import logging
import traceback
import os
import json
import base64
import math
import hashlib
import chardet
import gzip
import secrets
import uuid
from typing import Dict, Any, Optional, List
from functools import wraps
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

class Config:
    API_VERSION = "2.1.0"
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # CORS - مش مهمة للتطبيق لكن للـ testing
    ALLOWED_ORIGINS = os.getenv(
        'ALLOWED_ORIGINS', 
        'http://localhost:3000,http://localhost:5173,http://localhost:5000'
    ).split(',')
    
    # Rate Limiting
    RATE_LIMIT = os.getenv('RATE_LIMIT', '20 per minute')
    RATE_LIMIT_CLEAN = os.getenv('RATE_LIMIT_CLEAN', '5 per minute')  # أقل للـ cleaning
    
    # Cache
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', 300))
    CACHE_THRESHOLD = 100
    
    # Data limits - مخصصة للموبايل
    MAX_SCATTER_POINTS = 50  # أقل من قبل لتوفير data
    MAX_SAMPLE_SIZE = 50
    MAX_CORRELATION_COLS = 8
    MAX_HISTOGRAM_BINS = 10
    MAX_CATEGORICAL_VALUES = 8
    MAX_SCATTER_PAIRS = 8  # أقل pairs
    
    # Processing
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    CHUNK_SIZE = 8192
    
    # API Keys - للتطبيق
    API_KEYS = {
        secrets.token_urlsafe(32): "android_app_v1",
        # أضف المفاتيح هنا أو من environment variable
    }
    
    # Add API keys from environment
    env_api_keys = os.getenv('API_KEYS', '')
    if env_api_keys:
        for key in env_api_keys.split(','):
            if key.strip():
                API_KEYS[key.strip()] = "android_app"

# ═══════════════════════════════════════════════════════════
# APP INITIALIZATION
# ═══════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
app.config['COMPRESS_MIN_SIZE'] = 500  # ضغط أي response أكبر من 500 bytes

# CORS
CORS(app, resources={
    r"/*": {
        "origins": Config.ALLOWED_ORIGINS,
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization", "X-API-Key"]
    }
})

# Compression
Compress(app)

# Rate Limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[Config.RATE_LIMIT]
)

# Caching
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': Config.CACHE_DEFAULT_TIMEOUT,
    'CACHE_THRESHOLD': Config.CACHE_THRESHOLD
})

# Logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════

def require_api_key(f):
    """Decorator للتحقق من API key"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({
                "error": {
                    "code": "AUTH_REQUIRED",
                    "message": "API key is required",
                    "user_action": "Please add X-API-Key header"
                }
            }), 401
        
        if api_key not in Config.API_KEYS:
            logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
            return jsonify({
                "error": {
                    "code": "AUTH_INVALID",
                    "message": "Invalid API key",
                    "user_action": "Please check your API key"
                }
            }), 401
        
        # إضافة معلومات عن الـ client للـ request
        request.client_type = Config.API_KEYS[api_key]
        return f(*args, **kwargs)
    
    return decorated_function

def get_request_id():
    """إنشاء request ID فريد"""
    return str(uuid.uuid4())

@app.before_request
def before_request():
    """إضافة request ID لكل request"""
    request.id = get_request_id()

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
    import csv
    try:
        sniffer = csv.Sniffer()
        separator = sniffer.sniff(sample_text, delimiters=[',', ';', '\t', '|']).delimiter
        return separator
    except:
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

def read_file_in_chunks(file_storage):
    """قراءة الملف على قطع لتوفير الذاكرة"""
    chunks = []
    total_size = 0
    while True:
        chunk = file_storage.read(Config.CHUNK_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
        total_size += len(chunk)
        
        # حماية إضافية من الملفات الكبيرة
        if total_size > Config.MAX_CONTENT_LENGTH:
            raise ValueError("File too large")
    
    return b''.join(chunks)

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

def compute_histogram_bins(series, num_bins=10):
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

def compute_scatter_data(s, max_points=50):
    """إنشاء scatter data - محسنة للموبايل"""
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

def compute_outlier_points(s, outliers_mask, max_points=30):
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

def compute_sample_values(s, max_sample=50):
    """حساب قيم عينة - أقل للموبايل"""
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
            "outlier_pct": safe_float(outlier_pct) or 0.0,
            "null_count": null_count,
            "count": int(s.count()),
        }
        
        if include_visualizations:
            result.update({
                "scatter_data": compute_scatter_data(s, Config.MAX_SCATTER_POINTS),
                "outlier_points": compute_outlier_points(s, outliers_mask),
                "sample_values": compute_sample_values(s, Config.MAX_SAMPLE_SIZE),
                "histogram_bins": compute_histogram_bins(s, Config.MAX_HISTOGRAM_BINS)
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

def build_scatter_pairs(df, numeric_cols, max_cols=6, max_pairs=8, max_points=50):
    """بناء scatter pairs - أقل للموبايل"""
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

def validate_dataframe(df):
    """التحقق من صحة البيانات"""
    if df.empty:
        raise ValueError("Empty dataframe - no data found")
    
    if df.shape[1] == 0:
        raise ValueError("No columns found in the file")
    
    if len(df.columns) != len(set(df.columns)):
        logger.warning("Duplicate column names found - will be renamed")
        df.columns = [f"{col}_{i}" if list(df.columns).count(col) > 1 else col 
                     for i, col in enumerate(df.columns)]
    
    return df

def analyze_dataframe(df, include_visualizations=True, detail_level='medium'):
    """تحليل DataFrame كامل مع مستويات تفاصيل مختلفة"""
    df = validate_dataframe(df)
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
    
    # تحديد مستوى التفاصيل
    if detail_level == 'low':
        # إحصائيات أساسية فقط
        numeric_summary = {}
        for col in numeric_cols[:5]:  # أول 5 أعمدة فقط
            numeric_summary[col] = compute_column_summary(
                numeric_df[col], col, False  # بدون visualizations
            )
    else:
        # إحصائيات كاملة
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
    
    # Scatter pairs حسب مستوى التفاصيل
    if detail_level == 'low':
        scatter_pairs = []
    else:
        scatter_pairs = build_scatter_pairs(
            df=df,
            numeric_cols=numeric_cols,
            max_cols=6,
            max_pairs=Config.MAX_SCATTER_PAIRS,
            max_points=Config.MAX_SCATTER_POINTS
        )
    
    result = {
        "api_version": Config.API_VERSION,
        "request_id": request.id if hasattr(request, 'id') else str(uuid.uuid4()),
        "total_rows": total_rows,
        "total_columns": total_columns,
        "columns": columns,
        "data_types": data_types,
        "duplicates": duplicates,
        "missing_values": missing_dict,
        "missing_pct": safe_float(missing_pct) or 0.0,
        "total_missing_values": total_missing,
        "numeric_cols": numeric_cols,
        "numeric_summary": numeric_summary,
        "categorical_cols": cat_cols,
        "cat_summary": cat_summary,
        "correlation": correlation,
        "scatter_pairs": scatter_pairs,
        "detail_level": detail_level
    }
    
    return result

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
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        stats_file = os.path.join(BASE_DIR, 'data', 'stats.json')
        
        with open(stats_file, 'r', encoding='utf-8') as f:
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
            "/health": "GET - Server health check",
            "/version": "GET - Check API version for updates"
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

@app.route('/version')
def check_version():
    """التحقق من إصدار API للتطبيق"""
    return jsonify({
        "api_version": Config.API_VERSION,
        "minimum_android_version": "1.0.0",
        "force_update": False,
        "update_message": None,
        "endpoints": {
            "analyze": "/analyze",
            "clean": "/clean",
            "stats": "/stats"
        }
    })

@app.route('/stats')
@cache.cached(timeout=Config.CACHE_DEFAULT_TIMEOUT)
@require_api_key
def stats():
    """Get demo statistics"""
    try:
        demo_stats = load_demo_stats()
        
        response = {
            "api_version": Config.API_VERSION,
            "request_id": request.id,
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
        return jsonify({
            "error": {
                "code": "STATS_ERROR",
                "message": "Failed to load statistics",
                "user_action": "Please try again later"
            }
        }), 500

@app.route('/analyze', methods=['POST'])
@limiter.limit(Config.RATE_LIMIT)
@require_api_key
def analyze():
    """Analyze uploaded CSV file"""
    t0 = time.time()
    
    try:
        if 'file' not in request.files:
            return jsonify({
                "error": {
                    "code": "NO_FILE",
                    "message": "No file uploaded",
                    "user_action": "Please upload a CSV file using the 'file' field"
                }
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "error": {
                    "code": "NO_FILE_SELECTED",
                    "message": "No file selected",
                    "user_action": "Please select a file to upload"
                }
            }), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                "error": {
                    "code": "INVALID_FILE_TYPE",
                    "message": "Invalid file type",
                    "user_action": "File must be a CSV file"
                }
            }), 400
        
        # الحصول على مستوى التفاصيل
        detail_level = request.args.get('detail', 'medium')  # low, medium, high
        
        # قراءة الملف
        content_bytes = read_file_in_chunks(file)
        
        file_hash = get_file_hash(content_bytes)
        cache_key = f"analysis_{file_hash}_{detail_level}"
        
        cached_result = cache.get(cache_key)
        if cached_result:
            cached_result['cached'] = True
            cached_result['message'] = "Result retrieved from cache"
            cached_result['request_id'] = request.id
            return jsonify(cached_result)
        
        try:
            df = read_csv_smart(content_bytes)
        except ValueError as e:
            logger.error(f"Failed to read CSV: {e}")
            return jsonify({
                "error": {
                    "code": "CSV_READ_ERROR",
                    "message": "The file could not be read",
                    "user_action": "Please check the file format"
                }
            }), 400
        
        include_viz = request.args.get('include_viz', 'true').lower() == 'true'
        
        result = analyze_dataframe(df, include_viz, detail_level)
        
        elapsed_ms = int((time.time() - t0) * 1000)
        result['processing_ms'] = elapsed_ms
        result['cached'] = False
        result['message'] = "File analyzed successfully"
        result['request_id'] = request.id
        
        # Cache النتيجة
        cache.set(cache_key, result, timeout=Config.CACHE_DEFAULT_TIMEOUT)
        
        # تسجيل العملية
        logger.info(f"Request {request.id}: File analyzed - {file.filename} - {total_rows} rows - {elapsed_ms}ms")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis failed for request {request.id}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": {
                "code": "ANALYSIS_FAILED",
                "message": "An error occurred while analyzing the file",
                "user_action": "Please try again later"
            }
        }), 500

@app.route('/clean', methods=['POST'])
@limiter.limit(Config.RATE_LIMIT_CLEAN)
@require_api_key
def clean():
    """Clean uploaded CSV file - returns downloadable file"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "error": {
                    "code": "NO_FILE",
                    "message": "No file uploaded",
                    "user_action": "Please upload a CSV file using the 'file' field"
                }
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "error": {
                    "code": "NO_FILE_SELECTED",
                    "message": "No file selected",
                    "user_action": "Please select a file to upload"
                }
            }), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                "error": {
                    "code": "INVALID_FILE_TYPE",
                    "message": "Invalid file type",
                    "user_action": "File must be a CSV file"
                }
            }), 400
        
        content_bytes = read_file_in_chunks(file)
        
        try:
            df = read_csv_smart(content_bytes)
        except ValueError as e:
            logger.error(f"Failed to read CSV: {e}")
            return jsonify({
                "error": {
                    "code": "CSV_READ_ERROR",
                    "message": "The file could not be read",
                    "user_action": "Please check the file format"
                }
            }), 400
        
        df_cleaned, before, after = clean_dataframe(df)
        
        # تحضير الملف للتنزيل
        csv_buffer = io.StringIO()
        df_cleaned.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_string = csv_buffer.getvalue()
        
        original_name = file.filename.rsplit('.', 1)[0]
        clean_filename = f"cleaned_{original_name}.csv"
        
        # إحصائيات التنظيف
        cleaning_stats = {
            "before": before,
            "after": after,
            "duplicates_removed": before['duplicates'],
            "missing_values_filled": before['missing'] - after['missing'],
            "rows_removed": before['rows'] - after['rows']
        }
        
        logger.info(f"Request {request.id}: File cleaned - {file.filename} - {cleaning_stats['rows_removed']} rows removed")
        
        # إرجاع الملف مباشرة مع الإحصائيات في headers
        response = send_file(
            io.BytesIO(csv_string.encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=clean_filename
        )
        
        # إضافة الإحصائيات في headers
        response.headers['X-Cleaning-Stats'] = json.dumps(cleaning_stats)
        response.headers['X-Request-ID'] = request.id
        
        return response
        
    except Exception as e:
        logger.error(f"Cleaning failed for request {request.id}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": {
                "code": "CLEANING_FAILED",
                "message": "An error occurred while cleaning the file",
                "user_action": "Please try again later"
            }
        }), 500

@app.route('/clean-stats', methods=['POST'])
@limiter.limit(Config.RATE_LIMIT_CLEAN)
@require_api_key
def clean_stats():
    """Clean CSV file and return only statistics (بدون تنزيل الملف)"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "error": {
                    "code": "NO_FILE",
                    "message": "No file uploaded"
                }
            }), 400
        
        file = request.files['file']
        content_bytes = read_file_in_chunks(file)
        
        try:
            df = read_csv_smart(content_bytes)
        except ValueError as e:
            return jsonify({
                "error": {
                    "code": "CSV_READ_ERROR",
                    "message": "The file could not be read"
                }
            }), 400
        
        df_cleaned, before, after = clean_dataframe(df)
        
        cleaning_stats = {
            "before": before,
            "after": after,
            "duplicates_removed": before['duplicates'],
            "missing_values_filled": before['missing'] - after['missing'],
            "rows_removed": before['rows'] - after['rows'],
            "request_id": request.id
        }
        
        return jsonify({
            "success": True,
            "message": "File cleaned successfully",
            "cleaning_stats": cleaning_stats
        })
        
    except Exception as e:
        logger.error(f"Cleaning stats failed for request {request.id}: {str(e)}")
        return jsonify({
            "error": {
                "code": "CLEANING_FAILED",
                "message": "An error occurred while cleaning the file"
            }
        }), 500

# ═══════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════

@app.errorhandler(400)
def bad_request(e):
    return jsonify({
        "error": {
            "code": "BAD_REQUEST",
            "message": "The request could not be understood",
            "user_action": "Please check your request and try again"
        }
    }), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": {
            "code": "NOT_FOUND",
            "message": "The requested endpoint does not exist",
            "user_action": "Please check the API documentation"
        }
    }), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": {
            "code": "FILE_TOO_LARGE",
            "message": f"Maximum file size is {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB",
            "user_action": "Please select a smaller file"
        }
    }), 413

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": {
            "code": "RATE_LIMIT_EXCEEDED",
            "message": "Too many requests",
            "user_action": "Please wait before making more requests",
            "retry_after": e.description if hasattr(e, 'description') else 60
        }
    }), 429

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({
        "error": {
            "code": "SERVER_ERROR",
            "message": "An unexpected error occurred",
            "user_action": "Please try again later"
        }
    }), 500

@app.errorhandler(401)
def unauthorized(e):
    return jsonify({
        "error": {
            "code": "UNAUTHORIZED",
            "message": "Authentication required",
            "user_action": "Please provide a valid API key"
        }
    }), 401

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    # طباعة معلومات التشغيل
    logger.info(f"Starting GoBike Analysis API v{Config.API_VERSION}")
    logger.info(f"Port: {port}")
    logger.info(f"Rate Limit: {Config.RATE_LIMIT}")
    logger.info(f"Max File Size: {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB")
    
    app.run(
        debug=False,
        host='0.0.0.0',
        port=port,
        threaded=True
    )