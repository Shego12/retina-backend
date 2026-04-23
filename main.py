from fastapi import FastAPI, File, UploadFile, Form
import cv2
import numpy as np
import io
from PIL import Image
from deepface import DeepFace
import sqlite3
import json
from datetime import datetime
import os

import firebase_admin
from firebase_admin import credentials, firestore

app = FastAPI(title="Retina Face Recognition API")

# --- ANTI-CRASH FIREBASE BOOTUP ---
db = None
try:
    firebase_env = os.environ.get("FIREBASE_JSON")
    
    if firebase_env and firebase_env.strip() != "":
        # Running on Railway
        cred_dict = json.loads(firebase_env)
        cred = credentials.Certificate(cred_dict)
    else:
        # Running Locally
        cred = credentials.Certificate("firebase-key.json")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ FIREBASE IS CONNECTED!")
except Exception as e:
    print("❌ FIREBASE INIT FAILED: Check your Railway Variable!")
    print(f"Error: {e}")

def get_db_connection():
    conn = sqlite3.connect("retina.db")
    conn.row_factory = sqlite3.Row
    return conn

def calculate_cosine_distance(source, test):
    a = np.matmul(np.transpose(source), test)
    b = np.sum(np.multiply(source, source))
    c = np.sum(np.multiply(test, test))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

@app.get("/")
def read_root():
    return {"message": "Retina API is live!"}

@app.post("/register")
async def register_face(uid: str = Form("unknown"), email: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    try:
        results = DeepFace.represent(img_path=frame_bgr, model_name="Facenet", enforce_detection=True)
        embedding = results[0]["embedding"]
        
        # 1. Convert to raw floats
        embedding_float = [float(v) for v in embedding]
        
        # 2. Create the text string ONLY for the local SQLite database
        embedding_json = json.dumps(embedding_float)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, face_encoding) VALUES (?, ?)", (email, embedding_json))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        # THE FIX: Sending raw arrays (embedding_float) to Web Dashboard, NOT text!
        if uid != "unknown" and db is not None:
            user_ref = db.collection("web_users").document(uid)
            user_ref.set({
                "faceEnrolled": True,
                "descriptors": [embedding_float],
                "averageDescriptor": embedding_float,
                "email": email,
                "employeeId": uid[:8].upper()
            }, merge=True)

        return {"status": "success", "message": f"Successfully registered face for '{email}'!", "user_id": user_id}

    except ValueError:
        return {"status": "error", "message": "No face detected. Please ensure good lighting."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/recognize")
async def recognize_face(email: str = Form(...), file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = DeepFace.represent(img_path=frame_bgr, model_name="Facenet", enforce_detection=True)
        new_embedding = results[0]["embedding"]

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, face_encoding FROM users")
        users = cursor.fetchall()

        best_match_name = "Unknown"
        best_match_id = None
        lowest_distance = 100 

        for user in users:
            db_embedding = json.loads(user["face_encoding"])
            distance = calculate_cosine_distance(db_embedding, new_embedding)

            if distance < 0.40 and distance < lowest_distance:
                lowest_distance = distance
                best_match_name = user["name"] 
                best_match_id = user["id"]

        if best_match_name != "Unknown":
            if best_match_name.lower() != email.lower():
                conn.close()
                return {"status": "error", "message": "Face does not match the logged-in account!"}

            now = datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            cooldown_seconds = 60 
            
            cursor.execute("SELECT timestamp FROM attendance_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (best_match_id,))
            last_log = cursor.fetchone()
            
            can_log = True
            if last_log:
                last_log_time = datetime.strptime(last_log[0], "%Y-%m-%d %H:%M:%S")
                time_difference = (now - last_log_time).total_seconds()
                if time_difference < cooldown_seconds:
                    can_log = False 

            if can_log:
                cursor.execute("INSERT INTO attendance_logs (user_id, timestamp) VALUES (?, ?)", (best_match_id, now_str))
                conn.commit()
                final_message = "Check-in logged successfully!"
                log_status = now_str
            else:
                final_message = "Welcome back!"
                log_status = "Skipped (Cooldown active)"

            conn.close()

            return {"status": "success", "message": final_message, "time_logged": log_status, "distance": round(lowest_distance, 3)}
        
        conn.close()
        return {"status": "error", "message": "Face not recognized in database."}

    except ValueError:
        return {"status": "error", "message": "No face detected."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.get("/logs")
def get_attendance_logs():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT attendance_logs.id, users.name, attendance_logs.timestamp FROM attendance_logs JOIN users ON attendance_logs.user_id = users.id ORDER BY attendance_logs.timestamp DESC")
        logs = cursor.fetchall()
        conn.close()

        log_list = [{"log_id": row["id"], "name": row["name"], "timestamp": row["timestamp"]} for row in logs]
        return {"status": "success", "total_logs": len(log_list), "data": log_list}

    except Exception as e:
        return {"status": "error", "message": str(e)}