from flask import Flask, render_template, request, jsonify, send_file, session, redirect
import os
from dotenv import load_dotenv
import requests
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from openai import OpenAI
from predict import DiseasePredictor

load_dotenv()   # this loads values from .env file

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

predictor = DiseasePredictor()
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["user"] = request.form["username"]
        return redirect("/dashboard")

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        return redirect("/login")

    return render_template("signup.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")

    history = session.get("history", [])
    return render_template("dashboard.html", user="vaish12", history=history)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty file"})

    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)

    
    result = predictor.predict(filepath)

    if "error" in result:
        return jsonify(result)

    
    weather = "Not available"
    try:
        res = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q=Hyderabad&appid={WEATHER_API_KEY}&units=metric"
        )
        data = res.json()
        weather = data["weather"][0]["description"]
    except:
        pass

    
    disease = result.get("disease", "Healthy Leaf")
    confidence = float(result.get("confidence", 0))

    description = result.get("description", "Fungal disease affecting leaves.")
    treatment = result.get("treatment", "Use fungicide spray.")
    fertilizer = result.get("fertilizer", "NPK fertilizer recommended.")

    
    if "history" not in session:
        session["history"] = []

    session["history"].append({
    "disease": disease,
    "confidence": confidence,
    "description": "Fungal disease affecting leaves.",
    "treatment": "Use fungicide spray.",
    "fertilizer": "NPK fertilizer recommended.",
    "weather": weather,
    "image_path": filepath
})

    session.modified = True

    
    return jsonify({
        "disease": disease,
        "confidence": round(confidence, 2),
        "description": description,
        "treatment": treatment,
        "fertilizer": fertilizer,
        "weather": weather
    })

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a plant disease assistant."},
                {"role": "user", "content": user_msg}
            ]
        )

        reply = response.choices[0].message.content

    except Exception as e:
        reply = f"Error: {str(e)}"

    return jsonify({"reply": reply})

@app.route("/download_report")
def download_report():
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from flask import session, send_file
    from datetime import datetime

    file_path = "report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []

    
    content.append(Paragraph("<b>🌿 Plant Disease Report</b>", styles["Title"]))
    content.append(Spacer(1, 10))

    
    user = session.get("user", "Unknown User")
    date = datetime.now().strftime("%d-%m-%Y %H:%M")

    content.append(Paragraph(f"<b>User:</b> {user}", styles["Normal"]))
    content.append(Paragraph(f"<b>Date:</b> {date}", styles["Normal"]))
    content.append(Spacer(1, 10))

    if "history" not in session or len(session["history"]) == 0:
        content.append(Paragraph("No data available.", styles["Normal"]))
    else:
        last = session["history"][-1]

        content.append(Paragraph(f"<b>Disease:</b> {last.get('disease','N/A')}", styles["Normal"]))
        content.append(Paragraph(f"<b>Confidence:</b> {last.get('confidence',0):.2f}%", styles["Normal"]))
        content.append(Paragraph(f"<b>Description:</b> {last.get('description','N/A')}", styles["Normal"]))
        content.append(Paragraph(f"<b>Treatment:</b> {last.get('treatment','N/A')}", styles["Normal"]))
        content.append(Paragraph(f"<b>Fertilizer:</b> {last.get('fertilizer','N/A')}", styles["Normal"]))
        content.append(Paragraph(f"<b>Weather:</b> {last.get('weather','N/A')}", styles["Normal"]))

    doc.build(content)

    return send_file(file_path, as_attachment=True)
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

if __name__ == "__main__":
    app.run(debug=True)