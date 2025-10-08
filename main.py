import os
import io
import traceback
import tempfile

import numpy as np
import soundfile as sf
import whisper
import joblib
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse

# -----------------------------
# Load Whisper model
# -----------------------------
whisper_model = whisper.load_model("base")  # or "tiny" for faster CPU

# -----------------------------
# Load TF-IDF Classifier
# -----------------------------
clf = joblib.load("tfidf_intent_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# Helper: Predict intent
# -----------------------------
def predict_intent(text: str):
    if not text.strip():  # empty transcription fallback
        return "Unknown"
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="🎤 Whisper + TF-IDF API", version="1.0")

@app.get("/")
async def serve_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse({"message": "Index file not found"}, status_code=404)

@app.post("/process/")
async def process_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        return JSONResponse({"error": "Only WAV files are supported"}, status_code=400)

    try:
        # Save uploaded WAV to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = tmp.name
            tmp.write(await file.read())

        # Read WAV file using soundfile (bypasses ffmpeg)
        audio, sr = sf.read(temp_path, dtype='float32')

        # Ensure mono for Whisper
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Transcribe using Whisper (no ffmpeg)
        result = whisper_model.transcribe(audio, fp16=False)
        transcription = result.get("text", "").strip()

        # Predict intent
        intent = predict_intent(transcription)

        # Cleanup temp file
        os.remove(temp_path)

        return {"transcription": transcription, "intent": intent}

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)
