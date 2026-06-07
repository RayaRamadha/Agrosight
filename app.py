"""
AgroSight — Flask Backend v2  (Production / Render)
Load model per kombinasi Komoditas × Provinsi dari folder models/
"""

import os
import json
import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ── App factory ──────────────────────────────────────────────
app = Flask(__name__)

# CORS: izinkan semua origin di production (atau ganti dengan domain spesifik)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Konstanta ────────────────────────────────────────────────
OIL_LAG_DAYS = 14
LAG_WINDOWS  = [1, 3, 7, 14]
MODELS_DIR   = os.environ.get("MODELS_DIR", "models")
LAST_PRICES  = os.environ.get("LAST_PRICES_FILE", "last_known_prices.json")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")

FEATURE_COLS = [f'lag_{l}' for l in LAG_WINDOWS] + [
    'day_of_year', 'month', 'week', 'day_of_week',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'oil_price', 'Oil_Lag_14',
]

KOMODITAS_LIST = [
    "Bawang Merah (kg)",
    "Bawang Putih Honan (kg)",
    "Beras Medium (kg)",
    "Beras Premium (kg)",
    "Cabai Merah Besar (kg)",
    "Cabai Merah Keriting (kg)",
    "Cabai Rawit Merah (kg)",
    "Daging Ayam Ras (kg)",
    "Daging Sapi Paha Belakang (kg)",
    "Gula Pasir Curah (kg)",
    "Kedelai Impor (kg)",
    "Minyak Goreng Sawit Curah (lt)",
    "Minyakita (lt)",
    "Telur Ayam Ras (kg)",
    "Tepung Terigu (kg)",
]

PROVINSI_LIST = [
    "Aceh", "Bali", "Banten", "Bengkulu", "DI Yogyakarta",
    "DKI Jakarta", "Gorontalo", "Jambi", "Jawa Barat", "Jawa Tengah",
    "Jawa Timur", "Kalimantan Barat", "Kalimantan Selatan",
    "Kalimantan Tengah", "Kalimantan Timur", "Kalimantan Utara",
    "Kepulauan Bangka Belitung", "Kepulauan Riau", "Lampung", "Maluku",
    "Maluku Utara", "Nusa Tenggara Barat", "Nusa Tenggara Timur",
    "Papua", "Papua Barat", "Riau", "Sulawesi Barat", "Sulawesi Selatan",
    "Sulawesi Tengah", "Sulawesi Tenggara", "Sulawesi Utara",
    "Sumatera Barat", "Sumatera Selatan", "Sumatera Utara",
]


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def load_last_prices() -> dict:
    if os.path.exists(LAST_PRICES):
        try:
            with open(LAST_PRICES) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def safe_key(komoditas: str, provinsi: str) -> str:
    return f"{komoditas}_{provinsi}".replace('/', '_').replace(' ', '_')


def load_model(komoditas: str, provinsi: str):
    path = os.path.join(MODELS_DIR, f"{safe_key(komoditas, provinsi)}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model untuk '{komoditas} — {provinsi}' belum tersedia. "
            f"Jalankan train_all_models.py terlebih dahulu."
        )
    return joblib.load(path)


# ── Oil price cache (TTL 1 jam) ──────────────────────────────
_oil_cache: dict = {'data': None, 'fetched_at': None}

def get_oil_data() -> pd.DataFrame:
    now = datetime.now()
    cached = _oil_cache
    if (
        cached['data'] is not None
        and cached['fetched_at'] is not None
        and (now - cached['fetched_at']).seconds < 3600
    ):
        return cached['data']

    oil_raw = yf.download(
        'BZ=F',
        start='2024-01-01',
        end=now.strftime('%Y-%m-%d'),
        progress=False,
        auto_adjust=True,
    )
    if isinstance(oil_raw.columns, pd.MultiIndex):
        oil_raw.columns = oil_raw.columns.get_level_values(0)

    oil = oil_raw[['Close']].rename(columns={'Close': 'oil_price'})
    oil = oil.resample('D').ffill()
    oil[f'Oil_Lag_{OIL_LAG_DAYS}'] = oil['oil_price'].shift(OIL_LAG_DAYS)
    oil = oil.reset_index()
    oil.rename(columns={oil.columns[0]: 'ds'}, inplace=True)
    oil['ds'] = pd.to_datetime(oil['ds'])
    oil.dropna(inplace=True)
    oil.reset_index(drop=True, inplace=True)

    cached['data']       = oil
    cached['fetched_at'] = now
    return oil


def time_features(dt: datetime) -> dict:
    month       = dt.month
    day_of_week = dt.weekday()
    return {
        'day_of_year': dt.timetuple().tm_yday,
        'month':        month,
        'week':         dt.isocalendar()[1],
        'day_of_week':  day_of_week,
        'month_sin':    np.sin(2 * np.pi * month       / 12),
        'month_cos':    np.cos(2 * np.pi * month       / 12),
        'dow_sin':      np.sin(2 * np.pi * day_of_week / 7),
        'dow_cos':      np.cos(2 * np.pi * day_of_week / 7),
    }


def recursive_forecast(
    model_bundle: dict,
    history: list,
    oil_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
) -> list:
    knn          = model_bundle['knn']
    scaler       = model_bundle['scaler']
    feature_cols = model_bundle['feature_cols']
    lag_col      = f'Oil_Lag_{OIL_LAG_DAYS}'

    forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    hist    = list(history)
    results = []

    for fdate in forecast_dates:
        lag_feats = {
            f'lag_{lag}': hist[-lag] if len(hist) >= lag else hist[0]
            for lag in LAG_WINDOWS
        }
        tf      = time_features(fdate.to_pydatetime())
        oil_row = oil_data[oil_data['ds'] <= pd.Timestamp(fdate)]

        if len(oil_row) > 0:
            oil_price = float(oil_row['oil_price'].iloc[-1])
            oil_lag   = float(oil_row[lag_col].iloc[-1])
        else:
            oil_price = float(oil_data['oil_price'].iloc[-1])
            oil_lag   = float(oil_data[lag_col].iloc[-1])

        row  = {**lag_feats, **tf, 'oil_price': oil_price, 'Oil_Lag_14': oil_lag}
        X    = np.array([[row[c] for c in feature_cols]])
        X_sc = scaler.transform(X)
        pred = float(knn.predict(X_sc)[0])
        pred = max(pred, 0)

        results.append({'date': fdate.strftime('%Y-%m-%d'), 'price': round(pred, 2)})
        hist.append(pred)

    return results


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def health():
    """Health-check endpoint — dipakai Render untuk verify service up."""
    return jsonify({'status': 'ok', 'service': 'AgroSight API', 'version': '2.0'})


@app.route('/api/options', methods=['GET'])
def get_options():
    return jsonify({
        'komoditas': KOMODITAS_LIST,
        'provinsi':  PROVINSI_LIST,
    })


@app.route('/api/forecast', methods=['POST'])
def forecast():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body kosong atau bukan JSON'}), 400

    komoditas = (data.get('komoditas') or '').strip()
    provinsi  = (data.get('provinsi')  or '').strip()
    end_date  = (data.get('end_date')  or '').strip()

    if not komoditas:
        return jsonify({'error': 'Field komoditas wajib diisi'}), 400
    if not provinsi:
        return jsonify({'error': 'Field provinsi wajib diisi'}), 400
    if not end_date:
        return jsonify({'error': 'Field end_date wajib diisi'}), 400

    try:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Format end_date harus YYYY-MM-DD'}), 400

    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    if end_dt <= today:
        return jsonify({'error': 'end_date harus merupakan tanggal di masa depan'}), 400

    # Batasi maksimum 365 hari ke depan
    max_end = today + timedelta(days=365)
    if end_dt > max_end:
        return jsonify({'error': 'end_date tidak boleh lebih dari 365 hari ke depan'}), 400

    try:
        model_bundle = load_model(komoditas, provinsi)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f'Gagal memuat model: {e}'}), 500

    try:
        oil_data = get_oil_data()
    except Exception as e:
        return jsonify({'error': f'Gagal mengambil data harga minyak: {e}'}), 500

    last_prices = load_last_prices()
    key         = safe_key(komoditas, provinsi)
    if key in last_prices and len(last_prices[key]) >= max(LAG_WINDOWS):
        history = last_prices[key]
    else:
        scaler  = model_bundle['scaler']
        history = [float(scaler.mean_[0])] * 20

    try:
        predictions = recursive_forecast(
            model_bundle=model_bundle,
            history=history,
            oil_data=oil_data,
            start_date=today + timedelta(days=1),
            end_date=end_dt,
        )
    except Exception as e:
        return jsonify({'error': f'Forecast gagal: {e}'}), 500

    return jsonify({
        'komoditas':   komoditas,
        'provinsi':    provinsi,
        'end_date':    end_date,
        'n_days':      len(predictions),
        'predictions': predictions,
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if not os.path.exists(METRICS_FILE):
        return jsonify({
            'error': 'metrics.json tidak ditemukan. Jalankan train_all_models.py terlebih dahulu.'
        }), 404

    try:
        with open(METRICS_FILE) as f:
            all_metrics = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return jsonify({'error': f'Gagal membaca metrics.json: {e}'}), 500

    komoditas = (request.args.get('komoditas') or '').strip()
    provinsi  = (request.args.get('provinsi')  or '').strip()

    if komoditas and provinsi:
        k = safe_key(komoditas, provinsi)
        m = all_metrics.get(k)
        if not m:
            return jsonify({'error': f'Tidak ada metrik untuk {komoditas} — {provinsi}'}), 404
        return jsonify(m)

    return jsonify(all_metrics)


# ── Error handlers ────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint tidak ditemukan'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method tidak diizinkan'}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ── Entrypoint ────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)
