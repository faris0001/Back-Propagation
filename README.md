# Prediksi Hasil Panen

Project Flask sederhana untuk memprediksi hasil panen menggunakan Artificial Neural Network.

## Struktur

- `app.py` - aplikasi Flask
- `templates/index.html` - tampilan web
- `requirements.txt` - dependensi Python
- `model.h5`, `scaler_X.save`, `scaler_y.save` - model dan scaler terlatih
- `static/loss.png` - grafik loss training

## Instalasi

1. Buat virtual environment:
   ```bash
   python -m venv venv
   .\\venv\\Scripts\\activate
   ```
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Jalankan aplikasi

```bash
python app.py
```

Buka browser di `http://127.0.0.1:5000`

## Catatan

- Model dilatih otomatis bila artefak belum ada.
- Input form sekarang mempertahankan nilai setelah submit.
- Error ditampilkan dengan jelas di halaman web.
