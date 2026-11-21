from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image
import sqlite3
import secrets
import hashlib
import os
import io
import tempfile
import threading

import torch
import torch.nn as nn
from torchvision import models, transforms


# =========================================================
# MEMORY OPTIMIZATION
# =========================================================

# Keep CPU memory usage low on Render Free
torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"

DB = Path(
    os.getenv(
        "DB_PATH",
        str(BASE_DIR / "deepfake.db")
    )
)

DB.parent.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "deepfake_mobilenetv3.pth"
)


# =========================================================
# LIMITS
# =========================================================

# Render Free memory optimization
MAX_BYTES = 10 * 1024 * 1024

# Maximum video frames to analyze
MAX_VIDEO_FRAMES = 8


# =========================================================
# MODEL VARIABLES
# =========================================================

DEVICE = torch.device("cpu")

MODEL = None
CHECKPOINT = None
MODEL_LOAD_ERROR = None

MODEL_LOCK = threading.Lock()

IMAGE_SIZE = 224
IMAGE_TRANSFORM = None

CLASS_MAP = {
    "real": 0,
    "fake": 1
}


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="DeepFAke AI",
    version="2.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATABASE
# =========================================================

def db():
    connection = sqlite3.connect(
        DB,
        check_same_thread=False
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_db():

    connection = db()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            media_type TEXT NOT NULL,
            verdict TEXT,
            confidence REAL,
            real_probability REAL,
            fake_probability REAL,
            faces INTEGER,
            frames INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()

    connection.close()


init_db()


# =========================================================
# AUTHENTICATION
# =========================================================

class AuthIn(BaseModel):
    email: EmailStr
    password: str


def hash_password(password: str) -> str:

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def token_from_header(
    authorization: str
):

    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        return None

    return authorization.replace(
        "Bearer ",
        "",
        1
    ).strip()


def get_user(token):

    if not token:
        return None

    connection = db()

    row = connection.execute(
        """
        SELECT users.*
        FROM sessions
        JOIN users
        ON users.id = sessions.user_id
        WHERE sessions.token = ?
        """,
        (token,)
    ).fetchone()

    connection.close()

    if row:
        return dict(row)

    return None


# =========================================================
# REGISTER
# =========================================================

@app.post("/api/register")
def register(data: AuthIn):

    email = str(data.email).lower().strip()

    if len(data.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 6 characters."
        )

    connection = db()

    try:

        cursor = connection.execute(
            """
            INSERT INTO users (
                email,
                password_hash,
                created_at
            )
            VALUES (?, ?, ?)
            """,
            (
                email,
                hash_password(data.password),
                datetime.now(
                    timezone.utc
                ).isoformat()
            )
        )

        connection.commit()

        user_id = cursor.lastrowid

    except sqlite3.IntegrityError:

        connection.close()

        raise HTTPException(
            status_code=409,
            detail="Email already registered."
        )

    connection.close()

    token = secrets.token_urlsafe(32)

    connection = db()

    connection.execute(
        """
        INSERT INTO sessions (
            token,
            user_id,
            created_at
        )
        VALUES (?, ?, ?)
        """,
        (
            token,
            user_id,
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    connection.commit()

    connection.close()

    return {
        "status": "success",
        "message": "Registration successful.",
        "token": token
    }


# =========================================================
# LOGIN
# =========================================================

@app.post("/api/login")
def login(data: AuthIn):

    email = str(data.email).lower().strip()

    connection = db()

    user = connection.execute(
        """
        SELECT *
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()

    connection.close()

    if not user:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    if user["password_hash"] != hash_password(
        data.password
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    token = secrets.token_urlsafe(32)

    connection = db()

    connection.execute(
        """
        INSERT INTO sessions (
            token,
            user_id,
            created_at
        )
        VALUES (?, ?, ?)
        """,
        (
            token,
            user["id"],
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    connection.commit()

    connection.close()

    return {
        "status": "success",
        "message": "Login successful.",
        "token": token
    }


# =========================================================
# CURRENT USER
# =========================================================

@app.get("/api/me")
def me(
    authorization: str = Header(default="")
):

    user = get_user(
        token_from_header(authorization)
    )

    if not user:

        raise HTTPException(
            status_code=401,
            detail="Not authenticated."
        )

    return {
        "id": user["id"],
        "email": user["email"],
        "created_at": user["created_at"]
    }


# =========================================================
# LOGOUT
# =========================================================

@app.post("/api/logout")
def logout(
    authorization: str = Header(default="")
):

    token = token_from_header(
        authorization
    )

    if token:

        connection = db()

        connection.execute(
            """
            DELETE FROM sessions
            WHERE token = ?
            """,
            (token,)
        )

        connection.commit()

        connection.close()

    return {
        "status": "success",
        "message": "Logged out successfully."
    }


# =========================================================
# SAVE SCAN
# =========================================================

def save_scan(
    user_id,
    filename,
    media_type,
    result
):

    connection = db()

    connection.execute(
        """
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
        """,
        (
            user_id,
            filename,
            media_type,
            result.get("verdict"),
            result.get("confidence"),
            result.get("real_probability"),
            result.get("fake_probability"),
            result.get("faces", 0),
            result.get("frames", 0),
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    connection.commit()

    connection.close()


# =========================================================
# MODEL LOADING
# =========================================================

def load_model():

    global MODEL
    global CHECKPOINT
    global MODEL_LOAD_ERROR
    global IMAGE_SIZE
    global IMAGE_TRANSFORM
    global CLASS_MAP

    try:

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Model checkpoint not found: {MODEL_PATH}"
            )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )

        if checkpoint.get(
            "architecture"
        ) != "mobilenet_v3_small":

            raise ValueError(
                "Unsupported model architecture."
            )

        class_map = checkpoint.get(
            "class_map",
            {}
        )

        if (
            "real" not in class_map
            or "fake" not in class_map
        ):

            raise ValueError(
                "Checkpoint has an invalid class map."
            )

        # Create MobileNetV3-Small
        model = models.mobilenet_v3_small(
            weights=None
        )

        in_features = (
            model.classifier[3].in_features
        )

        model.classifier[3] = nn.Linear(
            in_features,
            2
        )

        # Load trained weights
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        model.eval()

        model.to(DEVICE)

        # Image configuration
        IMAGE_SIZE = int(
            checkpoint.get(
                "image_size",
                224
            )
        )

        mean = checkpoint.get(
            "mean",
            [
                0.485,
                0.456,
                0.406
            ]
        )

        std = checkpoint.get(
            "std",
            [
                0.229,
                0.224,
                0.225
            ]
        )

        CLASS_MAP = {
            "real": int(
                class_map["real"]
            ),
            "fake": int(
                class_map["fake"]
            )
        }

        # Create transform only once
        IMAGE_TRANSFORM = transforms.Compose(
            [
                transforms.Resize(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean,
                    std=std
                )
            ]
        )

        MODEL = model

        # IMPORTANT:
        # Keep only lightweight checkpoint information.
        # Do NOT keep model_state_dict in RAM.
        CHECKPOINT = {
            "architecture":
                "mobilenet_v3_small",

            "image_size":
                IMAGE_SIZE,

            "mean":
                mean,

            "std":
                std,

            "class_map":
                CLASS_MAP,

            "best_val_accuracy":
                checkpoint.get(
                    "best_val_accuracy"
                )
        }

        # Free original checkpoint RAM
        del checkpoint

        MODEL_LOAD_ERROR = None

        print(
            "DeepFAke AI: trained MobileNetV3 model loaded."
        )

        print(
            "Class map:",
            CLASS_MAP
        )

        print(
            "Saved validation accuracy:",
            CHECKPOINT.get(
                "best_val_accuracy"
            )
        )

    except Exception as error:

        MODEL = None

        CHECKPOINT = None

        IMAGE_TRANSFORM = None

        MODEL_LOAD_ERROR = str(error)

        print(
            "MODEL LOAD ERROR:",
            MODEL_LOAD_ERROR
        )


load_model()


# =========================================================
# IMAGE TRANSFORM
# =========================================================

def get_transform():

    if IMAGE_TRANSFORM is None:

        raise RuntimeError(
            "Image transform is not initialized."
        )

    return IMAGE_TRANSFORM


# =========================================================
# IMAGE PREDICTION
# =========================================================

def predict_image(image):

    if MODEL is None:

        raise RuntimeError(
            f"Trained model unavailable: "
            f"{MODEL_LOAD_ERROR}"
        )

    image = image.convert("RGB")

    tensor = IMAGE_TRANSFORM(
        image
    ).unsqueeze(0)

    with MODEL_LOCK:

        with torch.inference_mode():

            output = MODEL(
                tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

    real_index = CLASS_MAP["real"]

    fake_index = CLASS_MAP["fake"]

    real_probability = float(
        probabilities[
            real_index
        ].item()
    )

    fake_probability = float(
        probabilities[
            fake_index
        ].item()
    )

    is_fake = (
        fake_probability
        >= real_probability
    )

    # Free temporary tensors
    del tensor
    del output
    del probabilities

    return {

        "verdict":
            (
                "Likely fake"
                if is_fake
                else "Likely authentic"
            ),

        "confidence":
            max(
                real_probability,
                fake_probability
            ),

        "real_probability":
            real_probability,

        "fake_probability":
            fake_probability,

        "predicted_class":
            (
                "fake"
                if is_fake
                else "real"
            )
    }


# =========================================================
# FACE DETECTION
# =========================================================

def count_faces(image):

    try:

        import cv2
        import numpy as np

        # Reduce image size before OpenCV processing
        small = image.copy()

        small.thumbnail(
            (
                800,
                800
            )
        )

        rgb = np.asarray(
            small.convert("RGB")
        )

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        )

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        detector = cv2.CascadeClassifier(
            cascade_path
        )

        if detector.empty():

            return 0

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        return len(faces)

    except Exception:

        return 0


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():

    return {

        "status":
            "ok",

        "service":
            "DeepFAke AI",

        "model_ready":
            MODEL is not None,

        "mode":
            (
                "trained_mobilenetv3"
                if MODEL
                else "model_unavailable"
            ),

        "device":
            str(DEVICE),

        "model_error":
            MODEL_LOAD_ERROR
    }


# =========================================================
# IMAGE DETECTION
# =========================================================

@app.post("/api/detect-image")
async def detect_image(

    file: UploadFile = File(...),

    authorization: str = Header(
        default=""
    )
):

    # Read only maximum allowed amount
    data = await file.read(
        MAX_BYTES + 1
    )

    if not data:

        raise HTTPException(
            status_code=400,
            detail="Empty image file."
        )

    if len(data) > MAX_BYTES:

        raise HTTPException(
            status_code=413,
            detail=(
                "File too large. "
                "Maximum image size is 10 MB."
            )
        )

    try:

        image = Image.open(
            io.BytesIO(data)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    try:

        prediction = predict_image(
            image
        )

    except Exception as error:

        del image
        del data

        raise HTTPException(
            status_code=503,
            detail=str(error)
        )

    faces = count_faces(
        image
    )

    result = {

        **prediction,

        "frames":
            1,

        "faces":
            faces
    }

    user = get_user(
        token_from_header(
            authorization
        )
    )

    save_scan(

        user["id"]
        if user
        else None,

        file.filename
        or "image",

        "image",

        result
    )

    response = {

        "status":
            "Complete",

        **result,

        "media_type":
            "image",

        "engine":
            "MobileNetV3-Small",

        "confidence_level":
            "MODEL SCORE - NOT CALIBRATED",

        "note":
            (
                "Prediction generated by "
                "the trained MobileNetV3 model. "
                "The score is not a calibrated "
                "probability or a guarantee "
                "of authenticity."
            )
    }

    # Free memory
    del image
    del data

    return response


# =========================================================
# VIDEO DETECTION
# =========================================================

@app.post("/api/detect-video")
async def detect_video(

    file: UploadFile = File(...),

    authorization: str = Header(
        default=""
    )
):

    filename = (
        file.filename
        or "video"
    )

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    allowed_extensions = {

        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".mpeg",
        ".mpg"
    }

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format."
            )
        )

    data = await file.read(
        MAX_BYTES + 1
    )

    if not data:

        raise HTTPException(
            status_code=400,
            detail="Empty video file."
        )

    if len(data) > MAX_BYTES:

        raise HTTPException(
            status_code=413,
            detail=(
                "File too large. "
                "Maximum video size is 10 MB."
            )
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(data)

            temp_path = temp_file.name

        del data

        import cv2

        capture = cv2.VideoCapture(
            temp_path
        )

        if not capture.isOpened():

            raise HTTPException(
                status_code=400,
                detail="Could not open video."
            )

        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        if fps <= 0:

            fps = 25.0

        duration = (
            total_frames / fps
            if total_frames > 0
            else 0
        )

        if total_frames <= 0:

            capture.release()

            raise HTTPException(
                status_code=400,
                detail="Video contains no readable frames."
            )

        sample_count = min(
            MAX_VIDEO_FRAMES,
            total_frames
        )

        if sample_count <= 0:

            capture.release()

            raise HTTPException(
                status_code=400,
                detail="No frames available for analysis."
            )

        if sample_count == 1:

            frame_indexes = [0]

        else:

            frame_indexes = [

                int(
                    i * (
                        total_frames - 1
                    ) / (
                        sample_count - 1
                    )
                )

                for i in range(
                    sample_count
                )
            ]

        real_scores = []

        fake_scores = []

        faces_total = 0

        frames_processed = 0

        for frame_index in frame_indexes:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index
            )

            success, frame = (
                capture.read()
            )

            if not success:

                continue

            try:

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                pil_image = Image.fromarray(
                    rgb
                )

                prediction = predict_image(
                    pil_image
                )

                real_scores.append(
                    prediction[
                        "real_probability"
                    ]
                )

                fake_scores.append(
                    prediction[
                        "fake_probability"
                    ]
                )

                faces_total += count_faces(
                    pil_image
                )

                frames_processed += 1

                del rgb
                del pil_image

            finally:

                del frame

        capture.release()

        if not real_scores:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not process any "
                    "video frames."
                )
            )

        real_probability = (
            sum(real_scores)
            / len(real_scores)
        )

        fake_probability = (
            sum(fake_scores)
            / len(fake_scores)
        )

        is_fake = (
            fake_probability
            >= real_probability
        )

        confidence = max(
            real_probability,
            fake_probability
        )

        result = {

            "verdict":
                (
                    "Likely fake"
                    if is_fake
                    else "Likely authentic"
                ),

            "confidence":
                confidence,

            "real_probability":
                real_probability,

            "fake_probability":
                fake_probability,

            "predicted_class":
                (
                    "fake"
                    if is_fake
                    else "real"
                ),

            "frames":
                frames_processed,

            "faces":
                faces_total
        }

        user = get_user(
            token_from_header(
                authorization
            )
        )

        save_scan(

            user["id"]
            if user
            else None,

            filename,

            "video",

            result
        )

        return {

            "status":
                "Complete",

            **result,

            "media_type":
                "video",

            "engine":
                "MobileNetV3-Small",

            "total_frames":
                total_frames,

            "fps":
                fps,

            "duration":
                duration,

            "confidence_level":
                "MODEL SCORE - NOT CALIBRATED",

            "note":
                (
                    "Prediction generated from "
                    "sampled video frames using "
                    "the trained MobileNetV3 model. "
                    "The score is not a calibrated "
                    "probability or a guarantee "
                    "of authenticity."
                )
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        if temp_path:

            try:

                os.unlink(
                    temp_path
                )

            except OSError:

                pass


# =========================================================
# FRONTEND
# =========================================================

app.mount(
    "/",
    StaticFiles(
        directory=str(
            FRONTEND_DIR
        ),
        html=True
    ),
    name="frontend"
)