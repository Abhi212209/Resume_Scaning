from flask import Flask, render_template, request, redirect, url_for
import os
import fitz  # PyMuPDF for PDF reading
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and vectorizer
model = joblib.load('job_predictor_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'resume' not in request.files:
        return "No file uploaded", 400
    file = request.files['resume']
    if file.filename == '':
        return "Empty filename", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Extract and vectorize resume text
    resume_text = extract_text_from_pdf(filepath)
    vectorized_text = vectorizer.transform([resume_text])
    
    # Predict job role
    prediction = model.predict(vectorized_text)[0]

    return render_template('result.html', role=prediction)

if __name__ == '__main__':
    app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

