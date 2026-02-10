from flask import Flask, render_template, redirect, url_for, session, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import os
import cv2
import numpy as np
import requests
import psutil

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")

load_dotenv()
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
DATA_PATH = os.getenv("KDD_DATA_PATH", "data/KDDTrain+.txt")

GEOLOCATION_SERVICE_URL = os.getenv("GEOLOCATION_SERVICE_URL", "http://ip-api.com/json/")
GEOLOCATION_API_KEY = os.getenv("GEOLOCATION_API_KEY")


def get_public_location():
    try:
        resp = requests.get(GEOLOCATION_SERVICE_URL, timeout=6)
        data = resp.json()
        if data.get("status") == "success":
            return {
                "lat": data["lat"],
                "lon": data["lon"],
                "city": data.get("city", ""),
                "region": data.get("regionName", ""),
                "country": data.get("country", "")
            }
    except:
        pass
    return None


# ------------------- EMAIL -------------------
def send_email_alert(features, timestamp, photo_path=None, map_link=None, location_summary=None, serious=False):
    if not (SENDER_EMAIL and SENDER_PASSWORD and RECEIVER_EMAIL):
        return

    subject = "🚨 Anomaly Detected in Laptop Activity"
    if serious:
        subject = "🚨🚨 SERIOUS ALERT: 3rd Anomaly Detected 🚨🚨"

    body_lines = [
        f"An anomaly was detected at {timestamp}.",
        "Feature Details:",
        f"Duration (seconds): {features['Duration (seconds)']}",
        f"Network Sent (KB): {features['Source Bytes']}",
        f"Network Received (KB): {features['Destination Bytes']}",
        f"CPU Usage (%): {features['Wrong Fragment']*10:.1f}",
        f"RAM Usage (%): {features['Urgent']*10:.1f}"
    ]

    if map_link:
        body_lines.append(f"Location: {location_summary}")
        body_lines.append(f"Google Maps: {map_link}")

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText("\n".join(body_lines), "plain"))

    if photo_path and os.path.exists(photo_path):
        with open(photo_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(photo_path)}")
        msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
    except:
        pass


# ------------------- CAMERA -------------------
def capture_photo(timestamp):
    try:
        os.makedirs("captured_images", exist_ok=True)
        safe_ts = timestamp.replace(":", "-").replace(" ", "_")
        file_path = os.path.join("captured_images", f"anomaly_{safe_ts}.jpg")

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return None
        ret, frame = cam.read()
        if ret:
            cv2.imwrite(file_path, frame)
        cam.release()
        return file_path
    except:
        return None


# ------------------- FEATURES -------------------
def collect_system_features():
    duration = 1
    src_bytes = psutil.net_io_counters().bytes_sent / 10
    dst_bytes = psutil.net_io_counters().bytes_recv / 10
    wrong_fragment = psutil.cpu_percent(interval=1) * 10
    urgent = psutil.virtual_memory().percent * 10

    return {
        "Duration (seconds)": duration,
        "Source Bytes": src_bytes,
        "Destination Bytes": dst_bytes,
        "Wrong Fragment": wrong_fragment,
        "Urgent": urgent
    }


# ------------------- MODEL -------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
try:
    df = pd.read_csv(DATA_PATH, header=None)
    df[41] = df[41].apply(lambda x: 0 if str(x).strip() == "normal" else 1)
    X = df[[0, 4, 5, 7, 8]].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df[41]
    model.fit(X, y)
except:
    rng = np.random.RandomState(42)
    X_syn = rng.normal(size=(500, 5)) * [50, 1000, 1000, 1, 0.5] + [10, 500, 200, 0, 0]
    y_syn = rng.randint(0, 2, size=(500,))
    model.fit(X_syn, y_syn)


alerts = []
anomaly_counter = 0


# ------------------- ROUTES -------------------
@app.route('/')
def root():
    return redirect(url_for("login"))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        if request.form.get("username") == "akshitha" and request.form.get("password") == "akshi":
            session["user"] = "akshitha"
            return redirect(url_for("alert_page"))
        return render_template("login1.html", error="Invalid credentials")
    return render_template("login1.html")


@app.route('/logout')
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route('/alerts')
def alert_page():
    if "user" not in session:
        return redirect(url_for("login"))

    global anomaly_counter

    # -------- SINGLE CHECK Per Page Reload --------
    try:
        features = collect_system_features()
        sample = [[
            features["Duration (seconds)"],
            features["Source Bytes"],
            features["Destination Bytes"],
            features["Wrong Fragment"],
            features["Urgent"]
        ]]

        prob = model.predict_proba(sample)[0][1]
        prediction = 1 if prob > 0.35 else 0

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "🚨 Anomaly Detected" if prediction == 1 else "✅ Normal Activity"

        # SHOW both Normal + Anomaly in page
        alerts.append({
            "status": status,
            "time": timestamp,
            "features": features
        })

        # -------- EMAIL only if anomaly --------
        if prediction == 1:
            anomaly_counter += 1
            photo_path = capture_photo(timestamp)
            serious = (anomaly_counter % 3 == 0)

            map_link = location_summary = None
            if serious:
                loc = get_public_location()
                if loc:
                    map_link = f"https://www.google.com/maps/search/?api=1&query={loc['lat']},{loc['lon']}"
                    location_summary = f"{loc['city']}, {loc['region']}, {loc['country']}"

            send_email_alert(features, timestamp, photo_path, map_link, location_summary, serious)

    except Exception as e:
        app.logger.error(f"Error detecting anomaly: {e}")

    return render_template("f_alert.html", alerts=alerts)


@app.route('/verify_alert/<int:index>', methods=['POST'])
def verify_alert(index):
    if 0 <= index < len(alerts):
        alerts.pop(index)
    return redirect(url_for("alert_page"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
