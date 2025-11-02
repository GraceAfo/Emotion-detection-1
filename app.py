# app.py
# -----------------------------
# Flask web app for emotion detection
# -----------------------------

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sqlite3
import os
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model
model = load_model('face_emotionModel.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize database
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  email TEXT,
                  emotion TEXT,
                  image_path TEXT,
                  created_at TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Encouragement messages
def get_message(emotion):
    messages = {
        'Angry': "Take a deep breath — everything will be alright 💛",
        'Disgust': "Let go of what doesn't feel right 🌿",
        'Fear': "Courage doesn’t mean you’re not afraid — you’re stronger than you think 💪",
        'Happy': "Keep smiling! You light up the world 🌞",
        'Sad': "Cheer up, brighter days are coming 🌈",
        'Surprise': "Life’s full of wonders — embrace them 🎉",
        'Neutral': "Stay calm and keep your balance 🌸"
    }
    return messages.get(emotion, "You're doing great today!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    name = request.form['name']
    email = request.form['email']
    file = request.files['image']

    if not file:
        return "No image uploaded"

    filename = secure_filename(file.filename)
    os.makedirs('static', exist_ok=True)
    filepath = os.path.join('static', filename)
    file.save(filepath)

    # Convert image to 48x48 grayscale
    img = image.load_img(filepath, color_mode='grayscale', target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict emotion
    prediction = model.predict(img_array)
    emotion = emotion_labels[np.argmax(prediction)]
    message = get_message(emotion)

    # Save to DB
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (name, email, emotion, image_path, created_at) VALUES (?, ?, ?, ?, ?)",
              (name, email, emotion, filepath, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return f"""
    <h2>Detected Emotion: {emotion}</h2>
    <p>{message}</p>
    <br><img src='/{filepath}' width='200'>
    """

if __name__ == "__main__":
    print("✅ Flask app ready. Run locally using:")
    print("   python app.py")

