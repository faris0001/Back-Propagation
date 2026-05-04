from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

MODEL_PATH = 'model.h5'
SCALER_X_PATH = 'scaler_X.save'
SCALER_Y_PATH = 'scaler_y.save'
STATIC_DIR = 'static'
LOSS_PLOT = os.path.join(STATIC_DIR, 'loss.png')

# =========================
# DATASET
# =========================
def load_dataset():
    data = {
        'Curah_Hujan': [120, 135, 110, 145, 130, 140, 125, 150, 115, 155],
        'Suhu': [25.5, 26.1, 24.8, 27.0, 25.0, 26.5, 24.5, 27.5, 23.8, 28.0],
        'Kelembaban': [80, 78, 82, 75, 79, 77, 81, 74, 83, 73],
        'Nutrisi': [3.5, 3.8, 3.2, 4.0, 3.6, 3.9, 3.4, 4.2, 3.1, 4.3],
        'Penyinaran': [6, 7, 5, 8, 6.5, 7.5, 5.5, 9, 4.5, 9.5],
        'Hasil': [500, 550, 480, 600, 530, 570, 490, 620, 460, 630]
    }

    df = pd.DataFrame(data)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    return X, y


def ensure_static_folder():
    os.makedirs(STATIC_DIR, exist_ok=True)


def save_loss_plot(history):
    ensure_static_folder()
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Loss')
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT)
    plt.close()


# =========================
# TRAIN MODEL
# =========================
def train_model():
    X, y = load_dataset()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    model = Sequential([
        Dense(16, activation='relu', input_shape=(5,)),
        Dense(12, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_scaled, y_scaled, epochs=200, verbose=0)

    model.save(MODEL_PATH)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    save_loss_plot(history)

    return model, scaler_X, scaler_y


def load_artifacts():
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='mse')
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    return model, scaler_X, scaler_y


def artifacts_exist():
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH)


if artifacts_exist():
    try:
        model, scaler_X, scaler_y = load_artifacts()
    except Exception:
        model, scaler_X, scaler_y = train_model()
else:
    model, scaler_X, scaler_y = train_model()


# =========================
# UTILITIES
# =========================

def parse_float(field_name):
    value = request.form.get(field_name, '').strip()
    if value == '':
        raise ValueError(f'Nilai {field_name} harus diisi.')
    return float(value)


# =========================
# ROUTES
# =========================
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', hasil=None, error=None, form_data={})


@app.route('/predict', methods=['POST'])
def predict():
    error = None
    hasil = None
    form_data = {}

    try:
        hujan = parse_float('hujan')
        suhu = parse_float('suhu')
        lembab = parse_float('lembab')
        nutrisi = parse_float('nutrisi')
        cahaya = parse_float('cahaya')

        form_data = {
            'hujan': hujan,
            'suhu': suhu,
            'lembab': lembab,
            'nutrisi': nutrisi,
            'cahaya': cahaya
        }

        data = np.array([[hujan, suhu, lembab, nutrisi, cahaya]], dtype=float)
        data_scaled = scaler_X.transform(data)
        pred_scaled = model.predict(data_scaled, verbose=0)
        hasil = float(scaler_y.inverse_transform(pred_scaled)[0, 0])

    except ValueError as exc:
        error = str(exc)
    except Exception as exc:
        error = 'Terjadi kesalahan saat memproses prediksi. Pastikan input valid.'
        app.logger.exception(exc)

    return render_template('index.html', hasil=round(hasil, 2) if hasil is not None else None, error=error, form_data=form_data)


# =========================
# MAIN
# =========================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
