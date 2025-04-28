from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="Poverty Level Prediction API")

# Load model dan scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Skema input
class PovertyData(BaseModel):
    Provinsi: str
    Kab_Kota: str
    Persentase_Miskin: float
    Rata2_Lama_Sekolah: float
    Pengeluaran_per_Kapita: float 
    IPM: float
    Umur_Harapan_Hidup: float
    Akses_Sanitasi: float
    Akses_Air_Minum: float
    Tingkat_Pengangguran: float  
    TPAK: float
    PDRB: float 
    categorize_wilayah: int

def categorize_wilayah(ipm):
    if ipm < 60:
        return 1  # rentan
    else:
        return 0  # tidak rentan

def preprocess_input(data: PovertyData):
    # Buat DataFrame dari input
    df = pd.DataFrame([{
        "Persentase_Miskin": data.Persentase_Miskin,
        "Rata2_Lama_Sekolah": data.Rata2_Lama_Sekolah,
        "Pengeluaran_per_Kapita": data.Pengeluaran_per_Kapita,
        "IPM": data.IPM,
        "Umur_Harapan_Hidup": data.Umur_Harapan_Hidup,
        "Akses_Sanitasi": data.Akses_Sanitasi,
        "Akses_Air_Minum": data.Akses_Air_Minum,
        "Tingkat_Pengangguran": data.Tingkat_Pengangguran,  
        "TPAK": data.TPAK,
        "PDRB": data.PDRB, 
        "Wilayah_Rentan": data.categorize_wilayah
    }])

    # Tambahkan kolom Wilayah_Rentan berdasarkan IPM
    df['Wilayah_Rentan'] = df['IPM'].apply(categorize_wilayah)

    # Normalisasi data
    df_scaled = scaler.transform(df[[
        "Persentase_Miskin", "Rata2_Lama_Sekolah","Pengeluaran_per_Kapita", "IPM", "Umur_Harapan_Hidup",
        "Akses_Sanitasi", "Akses_Air_Minum", "Tingkat_Pengangguran", "TPAK",
        "PDRB", "Wilayah_Rentan"  # Pastikan semua fitur sesuai
    ]])
    return df_scaled

@app.get("/")
def read_root():
    return {"message": "Poverty Level Prediction API is running"}

# Endpoint prediksi
@app.post("/predict")
def predict_poverty(data: PovertyData):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]

    # Konversi prediksi ke kategori kemiskinan
    if prediction == 0:
        result = "Rendah"
    elif prediction == 1:
        result = "Sedang"
    else:
        result = "Tinggi"

    return {
        "Provinsi": data.Provinsi,
        "Kab/Kota": data.Kab_Kota,
        "Tingkat Kemiskinan": result
    }