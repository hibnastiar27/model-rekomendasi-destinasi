from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle

app = FastAPI()

list_tipe_destinasi = [
      'Tujuan Wisata', 'Monumen', 'Pantai', 'Toko Suvenir','Taman bermain', 'Patung', 'Toko Pie', 
      'Bangunan Bersejarah','Taman Rekreasi Air', 'Toko Roti', 'Titik Pemandangan','Taman Hiburan', 
      'Benteng', 'Museum Sejarah', 'Taman','Pembuat monumen', 'Arena Bermain Anak-Anak', 'Puncak Gunung',
      'Bangunan Terkenal', 'Museum', 'Kebun Binatang', 'Area Mendaki','Pusat Perbelanjaan', 'Pusat Rekreasi',
      'Museum arkeologi','Gunung berapi', 'Wahana Taman Hiburan','Taman Bermain Dalam Ruangan', 
      'Taman Peringatan','Bumi perkemahan', 'Area Rekreasi Alam', 'Taman Kota','Pusat Hiburan', 'Peternakan', 
      'Pusat Hiburan Anak-Anak','Taman Komunitas', 'Museum Ilmu Pengetahuan Alam', 'Promenade','Pantai Umum', 
      'Taman Margasatwa dan Safari', 'Hutan Nasional','Taman Ekologi', 'Museum Seni', 'Toko Kue', 'Cagar Alam',
      'Museum Rel Kereta', 'Semenanjung', 'Museum Bahari','Tempat bermain gokart', 'Arsip Negara',
      'Museum Angkatan Bersenjata', 'Museum Patung', 'Museum tempat bersejarah', 'Museum Sejarah Lokal', 
      'Lahan Piknik','Kebun Raya', 'Situs purbakala', 'Museum Hewan', 'Tempat Bersejarah', 'Taman Nasional', 
      'Perlindungan Margasatwa', 'Akuarium', 'Perpustakaan', 'Museum Seni Modern',
      'Wahana Bermain Salju Dalam Ruangan', 'Pusat kebudayaan','Museum Pusaka', 'Universitas Negeri', 
      'Galeri Seni','Museum Nasional', 'Warung Camilan', 'Complex volcano', 'Kastel','Museum Sains', 
      'Rumah Berhantu', 'Peternakan wisata', 'Teater Seni Pertunjukan', 'Pelestarian Situs Peninggalan',
       'Klub Pecinta Sejarah', 'Taman karavan','Taman bermain papan seluncur']
  # [daftar seperti sebelumnya]

model = None
vector_user = None
vector_dest = None

class UserInput(BaseModel):
    user_survey: str

def load_model(path: str):
    try:
        return tf.saved_model.load(path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_vocab(path: str):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading vocab: {e}")
        return None

@app.on_event("startup")
def load_all_models():
    global model, vector_user, vector_dest
    print("Loading model dan vectorizer...")

    model = load_model("./models/model_recommendation")
    vector_user = load_vocab('./models/vectorizer_user.pkl')
    vector_dest = load_vocab('./models/vectorizer_dest.pkl')

    print("Model loaded:", model is not None)
    print("Vectorizer user loaded:", vector_user is not None)
    print("Vectorizer dest loaded:", vector_dest is not None)

@app.post('/recommendation')
def recommendation(data: UserInput, max_recom: int = Query(5), treshold: float = Query(0.5)):
    try:
        if not (model and vector_user and vector_dest):
            return {
                "status": "error",
                "message": "Model atau vocab belum berhasil di-load",
                "data": []
            }

        main_function = model.signatures['serving_default']
        user_input = data.user_survey
        destinasi_input = list_tipe_destinasi

        user_input_array = np.array([user_input] * len(destinasi_input), dtype=object)
        destinasi_input_array = np.array(destinasi_input, dtype=object)

        user_input_token = tf.cast(vector_user(user_input_array), tf.int32)
        destinasi_input_token = tf.cast(vector_dest(destinasi_input_array), tf.int32)

        output = main_function(user_preferensi=user_input_token, tipe_destinasi=destinasi_input_token)
        predictions = output[next(iter(output))].numpy()

        hasil = list(zip(destinasi_input, predictions))
        hasil_urut = sorted(hasil, key=lambda x: x[1], reverse=True)

        hasil_json = [
            {"tipe_destinasi": tipe, "score": round(float(score), 2)}
            for tipe, score in hasil_urut if score > treshold
        ][:max_recom]

        return {
            "status": "success",
            "message": "Rekomendasi berhasil dibuat",
            "data": hasil_json
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Gagal melakukan rekomendasi: {e}",
            "data": []
        }

@app.get("/")
def read_root():
    return FileResponse("templates/rekomendasi.html")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
