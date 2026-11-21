
import os
import io
import cv2
import json
import uuid
import hashlib
import sqlite3
import tempfile
import numpy as np
import onnxruntime as ort

from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Header
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "deepfake_mobilenetv3.onnx"
CONFIG_PATH = BASE_DIR / "models" / "deepfake_mobilenetv3_config.json"

DB = Path(
    os.environ.get(
        "DB_PATH",
        str(BASE_DIR / "deepfake.db")
    )
)

FRONTEND_DIR = BASE_DIR / "frontend"

MAX_IMAGE_SIZE = 15 * 1024 * 1024
MAX_VIDEO_SIZE = 100 * 1024 * 1024

MODEL = None
MODEL_CONFIG = None

CLASS_MAP = {
    "real": 0,
    "fake": 1
}

IMAGE_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# =========================================================
# 2. FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="DeepFAke AI",
    description="AI Based Deepfake Detection System",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# 3. DATABASE CONNECTION
# =========================================================

def get_connection():

    conn = sqlite3.connect(
        str(DB),
        timeout=30
    )

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA busy_timeout = 30000")

    return conn


# =========================================================
# 4. DATABASE INITIALIZATION AND MIGRATION
# =========================================================

def init_database():

    conn = get_connection()

    try:

        conn.execute("PRAGMA journal_mode = WAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                filename TEXT,
                media_type TEXT,
                verdict TEXT,
                confidence REAL,
                real_probability REAL,
                fake_probability REAL,
                faces INTEGER DEFAULT 0,
                frames INTEGER DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        # Add missing columns to an existing scans table.
        existing_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(scans)"
            ).fetchall()
        }

        migrations = {
            "user_id": "INTEGER",
            "filename": "TEXT",
            "media_type": "TEXT",
            "verdict": "TEXT",
            "confidence": "REAL",
            "real_probability": "REAL",
            "fake_probability": "REAL",
            "faces": "INTEGER DEFAULT 0",
            "frames": "INTEGER DEFAULT 0",
            "created_at": "TEXT"
        }

        for column, definition in migrations.items():

            if column not in existing_columns:

                conn.execute(
                    f"ALTER TABLE scans ADD COLUMN {column} {definition}"
                )

        # Fill missing timestamps in old records.
        conn.execute("""
            UPDATE scans
            SET created_at = ?
            WHERE created_at IS NULL OR created_at = ''
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ))

        conn.commit()

        print("SQLite database initialized successfully.")

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()


init_database()


# =========================================================
# 5. AUTHENTICATION HELPERS
# =========================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def create_session(user_id):

    token = str(uuid.uuid4())

    conn = get_connection()

    try:

        conn.execute("""
            INSERT INTO sessions (token, user_id, created_at)
            VALUES (?, ?, ?)
        """, (
            token,
            user_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()

    finally:

        conn.close()

    return token


def get_user(token):

    if not token:
        return None

    conn = get_connection()

    try:

        row = conn.execute("""
            SELECT users.id, users.email
            FROM users
            JOIN sessions
            ON users.id = sessions.user_id
            WHERE sessions.token = ?
        """, (token,)).fetchone()

        return dict(row) if row else None

    finally:

        conn.close()


def get_current_user(authorization):

    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        return None

    token = authorization.replace(
        "Bearer ", "", 1
    ).strip()

    return get_user(token)


# =========================================================
# 6. SAVE SCAN
# =========================================================

def save_scan(
    user_id,
    filename,
    media_type,
    verdict,
    confidence,
    real_probability,
    fake_probability,
    faces=0,
    frames=0
):

    if not user_id:
        return

    conn = get_connection()

    try:

        now = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        conn.execute("""
            INSERT INTO scans (
                user_id,
                filename,
                media_type,
                verdict,
                confidence,
                real_probability,
                fake_probability,
                faces,
                frames,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            filename,
            media_type,
            verdict,
            float(confidence),
            float(real_probability),
            float(fake_probability),
            int(faces),
            int(frames),
            now
        ))

        conn.commit()

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()


# =========================================================
# 7. LOAD ONNX MODEL
# =========================================================

def load_model():

    global MODEL
    global MODEL_CONFIG
    global CLASS_MAP
    global IMAGE_SIZE
    global MEAN
    global STD

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"ONNX model missing: {MODEL_PATH}"
        )

    if not CONFIG_PATH.exists():

        raise FileNotFoundError(
            f"Model config missing: {CONFIG_PATH}"
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        MODEL_CONFIG = json.load(file)

    IMAGE_SIZE = int(
        MODEL_CONFIG.get("image_size", 224)
    )

    MEAN = MODEL_CONFIG.get(
        "mean",
        [0.485, 0.456, 0.406]
    )

    STD = MODEL_CONFIG.get(
        "std",
        [0.229, 0.224, 0.225]
    )

    CLASS_MAP = MODEL_CONFIG.get(
        "class_map",
        {"real": 0, "fake": 1}
    )

    options = ort.SessionOptions()

    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    MODEL = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    print("DeepFAke AI: trained ONNX MobileNetV3 model loaded.")
    print("Class map:", CLASS_MAP)
    print("Image size:", IMAGE_SIZE)

    print(
        "Saved validation accuracy:",
        MODEL_CONFIG.get("best_val_accuracy")
    )

    print(
        "ONNX Runtime provider:",
        MODEL.get_providers()
    )


# =========================================================
# 8. IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    image = ImageOps.exif_transpose(image)

    image = image.convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    mean = np.asarray(
        MEAN,
        dtype=np.float32
    ).reshape(1, 1, 3)

    std = np.asarray(
        STD,
        dtype=np.float32
    ).reshape(1, 1, 3)

    image_array = (
        image_array - mean
    ) / std

    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array.astype(np.float32)


# =========================================================
# 9. AI PREDICTION
# =========================================================

def predict_image(image):

    if MODEL is None:

        raise HTTPException(
            status_code=503,
            detail="AI model is not loaded"
        )

    input_data = preprocess_image(image)

    input_name = MODEL.get_inputs()[0].name

    output = MODEL.run(
        None,
        {input_name: input_data}
    )[0]

    logits = np.asarray(
        output,
        dtype=np.float32
    ).reshape(-1)

    if len(logits) != 2:

        raise RuntimeError(
            "Model must return two class scores"
        )

    logits = logits - np.max(logits)

    probabilities = np.exp(logits)

    probabilities = (
        probabilities / np.sum(probabilities)
    )

    real_index = CLASS_MAP["real"]
    fake_index = CLASS_MAP["fake"]

    real_probability = float(
        probabilities[real_index] * 100
    )

    fake_probability = float(
        probabilities[fake_index] * 100
    )

    if fake_probability >= real_probability:

        verdict = "LIKELY FAKE"
        predicted_class = "fake"
        confidence = fake_probability

    else:

        verdict = "LIKELY REAL"
        predicted_class = "real"
        confidence = real_probability

    if confidence >= 80:
        confidence_level = "High"

    elif confidence >= 65:
        confidence_level = "Moderate"

    else:
        confidence_level = "Low"

    return {
        "verdict": verdict,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "real_probability": round(real_probability, 2),
        "fake_probability": round(fake_probability, 2),
        "confidence_level": confidence_level
    }


# =========================================================
# 10. FACE DETECTION
# =========================================================

def detect_faces(image):

    try:

        if not hasattr(cv2, "CascadeClassifier"):
            return 0

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        detector = cv2.CascadeClassifier(
            cascade_path
        )

        if detector.empty():
            return 0

        image_array = np.asarray(
            image.convert("RGB")
        )

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return len(faces)

    except Exception as error:

        print("Face detection warning:", error)

        return 0


# =========================================================
# 11. REGISTER
# =========================================================

@app.post("/api/register")
async def register(data: dict):

    email = str(
        data.get("email", "")
    ).strip().lower()

    password = str(
        data.get("password", "")
    )

    if not email or not password:

        raise HTTPException(
            status_code=400,
            detail="Email and password are required"
        )

    if len(password) < 6:

        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 6 characters"
        )

    conn = get_connection()

    try:

        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        if existing:

            raise HTTPException(
                status_code=409,
                detail="Account already exists"
            )

        cursor = conn.execute("""
            INSERT INTO users (
                email,
                password_hash,
                created_at
            )
            VALUES (?, ?, ?)
        """, (
            email,
            hash_password(password),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        user_id = cursor.lastrowid

        conn.commit()

    finally:

        conn.close()

    token = create_session(user_id)

    return {
        "status": "success",
        "message": "Registration successful",
        "token": token
    }


# =========================================================
# 12. LOGIN
# =========================================================

@app.post("/api/login")
async def login(data: dict):

    email = str(
        data.get("email", "")
    ).strip().lower()

    password = str(
        data.get("password", "")
    )

    conn = get_connection()

    try:

        user = conn.execute("""
            SELECT id, email, password_hash
            FROM users
            WHERE email = ?
        """, (email,)).fetchone()

    finally:

        conn.close()

    if not user:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    if user["password_hash"] != hash_password(password):

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    token = create_session(user["id"])

    return {
        "status": "success",
        "message": "Login successful",
        "token": token
    }


# =========================================================
# 13. CURRENT USER
# =========================================================

@app.get("/api/me")
async def me(
    authorization: str = Header(default=None)
):

    user = get_current_user(authorization)

    if not user:

        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    return {
        "status": "success",
        "user": user
    }


# =========================================================
# 14. LOGOUT
# =========================================================

@app.post("/api/logout")
async def logout(
    authorization: str = Header(default=None)
):

    if not authorization or not authorization.startswith("Bearer "):

        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    token = authorization.replace(
        "Bearer ", "", 1
    ).strip()

    conn = get_connection()

    try:

        conn.execute(
            "DELETE FROM sessions WHERE token = ?",
            (token,)
        )

        conn.commit()

    finally:

        conn.close()

    return {
        "status": "success",
        "message": "Logged out successfully"
    }


# =========================================================
# 15. SCAN HISTORY
# =========================================================

@app.get("/api/history")
async def get_history(
    authorization: str = Header(default=None)
):

    user = get_current_user(authorization)

    if not user:

        raise HTTPException(
            status_code=401,
            detail="Please login to view history"
        )

    conn = get_connection()

    try:

        rows = conn.execute("""
            SELECT
                id,
                filename,
                media_type,
                verdict,
                confidence,
                real_probability,
                fake_probability,
                faces,
                frames,
                created_at
            FROM scans
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 50
        """, (user["id"],)).fetchall()

        history = [
            dict(row)
            for row in rows
        ]

    finally:

        conn.close()

    return {
        "status": "success",
        "history": history,
        "scans": history,
        "total": len(history)
    }


# =========================================================
# 16. IMAGE DETECTION
# =========================================================

@app.post("/api/detect-image")
async def detect_image(
    file: UploadFile = File(...),
    authorization: str = Header(default=None)
):

    allowed = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    )

    filename = file.filename or "uploaded_image"

    extension = Path(filename).suffix.lower()

    if extension not in allowed:

        raise HTTPException(
            status_code=400,
            detail="Unsupported image format"
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_IMAGE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Image size must be below 15 MB"
        )

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        image = ImageOps.exif_transpose(
            image
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    result = predict_image(image)

    faces = detect_faces(image)

    user = get_current_user(authorization)

    if user:

        save_scan(
            user_id=user["id"],
            filename=filename,
            media_type="image",
            verdict=result["verdict"],
            confidence=result["confidence"],
            real_probability=result["real_probability"],
            fake_probability=result["fake_probability"],
            faces=faces,
            frames=1
        )

    return {
        "status": "Complete",
        "filename": filename,
        "media_type": "image",
        "engine": "ONNX Runtime - MobileNetV3",
        **result,
        "faces": faces,
        "frames": 1,
        "note": "AI prediction is an estimate, not definitive proof."
    }


# =========================================================
# 17. VIDEO DETECTION
# =========================================================

@app.post("/api/detect-video")
async def detect_video(
    file: UploadFile = File(...),
    authorization: str = Header(default=None)
):

    allowed = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    )

    filename = file.filename or "uploaded_video"

    extension = Path(filename).suffix.lower()

    if extension not in allowed:

        raise HTTPException(
            status_code=400,
            detail="Unsupported video format"
        )

    video_bytes = await file.read()

    if not video_bytes:

        raise HTTPException(
            status_code=400,
            detail="Empty video file"
        )

    if len(video_bytes) > MAX_VIDEO_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Video size must be below 100 MB"
        )

    temp_path = None
    capture = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_file:

            temp_file.write(video_bytes)

            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)

        if not capture.isOpened():

            raise HTTPException(
                status_code=400,
                detail="Unable to read video"
            )

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:

            raise HTTPException(
                status_code=400,
                detail="Video contains no readable frames"
            )

        sample_count = min(
            8,
            total_frames
        )

        positions = np.linspace(
            0,
            total_frames - 1,
            sample_count,
            dtype=int
        )

        predictions = []
        face_count = 0
        processed_frames = 0

        for position in positions:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(position)
            )

            success, frame = capture.read()

            if not success:
                continue

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame_image = Image.fromarray(
                frame_rgb
            )

            result = predict_image(frame_image)

            predictions.append((
                result["real_probability"],
                result["fake_probability"]
            ))

            face_count += detect_faces(frame_image)

            processed_frames += 1

        capture.release()
        capture = None

        if not predictions:

            raise HTTPException(
                status_code=400,
                detail="Unable to process video frames"
            )

        real_probability = float(
            np.mean([
                item[0] for item in predictions
            ])
        )

        fake_probability = float(
            np.mean([
                item[1] for item in predictions
            ])
        )

        if fake_probability >= real_probability:

            verdict = "LIKELY FAKE"
            predicted_class = "fake"
            confidence = fake_probability

        else:

            verdict = "LIKELY REAL"
            predicted_class = "real"
            confidence = real_probability

        if confidence >= 80:
            confidence_level = "High"

        elif confidence >= 65:
            confidence_level = "Moderate"

        else:
            confidence_level = "Low"

        result = {
            "verdict": verdict,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "real_probability": round(real_probability, 2),
            "fake_probability": round(fake_probability, 2),
            "confidence_level": confidence_level
        }

        user = get_current_user(authorization)

        if user:

            save_scan(
                user_id=user["id"],
                filename=filename,
                media_type="video",
                verdict=verdict,
                confidence=confidence,
                real_probability=real_probability,
                fake_probability=fake_probability,
                faces=face_count,
                frames=processed_frames
            )

        return {
            "status": "Complete",
            "filename": filename,
            "media_type": "video",
            "engine": "ONNX Runtime - MobileNetV3",
            **result,
            "faces": face_count,
            "frames": processed_frames,
            "total_frames": total_frames,
            "note": "Video prediction is based on sampled frames."
        }

    finally:

        if capture is not None:
            capture.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# =========================================================
# 18. HEALTH CHECK
# =========================================================

@app.get("/api/health")
async def health():

    return {
        "status": "success",
        "message": "DeepFAke AI backend is running",
        "model_ready": MODEL is not None,
        "mode": "onnx_mobilenetv3",
        "provider": (
            MODEL.get_providers()
            if MODEL is not None
            else []
        ),
        "model_accuracy": (
            MODEL_CONFIG.get("best_val_accuracy")
            if MODEL_CONFIG
            else None
        )
    }


# =========================================================
# 19. LOAD MODEL
# =========================================================

load_model()


# =========================================================
# 20. FRONTEND
# =========================================================

if FRONTEND_DIR.exists():

    app.mount(
        "/",
        StaticFiles(
            directory=str(FRONTEND_DIR),
            html=True
        ),
        name="frontend"
    )

else:

    print(
        "WARNING: Frontend directory not found:",
        FRONTEND_DIR
    )
