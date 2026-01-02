import os
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

import main
sys.modules['__main__'] = main

app = FastAPI()


def add_custom_features(X):
    X = X.copy()
    X['Income_Per_Loan'] = X['ApplicantIncome'] / (X['LoanAmount'] + 1)
    X['ApplicantIncome'] = np.log1p(X['ApplicantIncome'])
    return X

setattr(sys.modules['__main__'], 'add_custom_features', add_custom_features)

def get_model_path(model_name):
    """Checks multiple locations for the model files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path1 = os.path.join(base_dir, "models", model_name)
    
    path2 = os.path.join(os.getcwd(), "models", model_name)
    
    if os.path.exists(path1):
        return path1
    return path2

LOG_PATH = get_model_path("loan_logistic_model.joblib")
TREE_PATH = get_model_path("loan_tree_model.joblib")

try:
    log_model = joblib.load(LOG_PATH)
    print(f"SUCCESS: Logistic Model loaded from {LOG_PATH}")
except Exception as e:
    print(f"ERROR: Logistic Model failed. Path: {LOG_PATH}. Reason: {e}")
    log_model = None

try:
    tree_model = joblib.load(TREE_PATH)
    print(f"SUCCESS: Tree Model loaded from {TREE_PATH}")
except Exception as e:
    print(f"ERROR: Tree Model failed. Path: {TREE_PATH}. Reason: {e}")
    tree_model = None
    
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;700;800&display=swap" rel="stylesheet">
        <style>body { font-family: 'Plus Jakarta Sans', sans-serif; }</style>
    </head>
    <body class="bg-slate-50 min-h-screen flex items-center justify-center p-6">
        <div class="max-w-xl w-full bg-white p-10 rounded-[40px] shadow-2xl border border-slate-100">
            <h1 class="text-3xl font-extrabold text-slate-900 mb-2">LoanPredictionAI <span class="text-indigo-600 italic">Pro</span></h1>
            <p class="text-slate-400 font-medium mb-8">Advanced Risk Intelligence Assessment</p>
            
    <form action="/predict" method="post" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label class="block text-[10px] font-bold text-slate-400 uppercase mb-2 ml-1 tracking-widest">Monthly Income ($)</label>
                    <input type="number" name="ApplicantIncome" placeholder="e.g. 5000" required class="w-full p-4 bg-slate-50 rounded-2xl outline-none focus:ring-2 focus:ring-indigo-500 border-none">
                </div>
                <div>
                    <label class="block text-[10px] font-bold text-slate-400 uppercase mb-2 ml-1 tracking-widest">Co-App Income ($)</label>
                    <input type="number" name="CoapplicantIncome" placeholder="e.g. 0" required class="w-full p-4 bg-slate-50 rounded-2xl outline-none focus:ring-2 focus:ring-indigo-500 border-none">
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label class="block text-[10px] font-bold text-slate-400 uppercase mb-2 ml-1 tracking-widest">Loan Request ($)</label>
                    <input type="number" name="LoanAmount" placeholder="e.g. 150" required class="w-full p-4 bg-slate-50 rounded-2xl outline-none focus:ring-2 focus:ring-indigo-500 border-none">
                </div>
                <div>
                    <label class="block text-[10px] font-bold text-slate-400 uppercase mb-2 ml-1 tracking-widest">Loan Term (Days)</label>
                    <select name="Loan_Amount_Term" class="w-full p-4 bg-slate-50 rounded-2xl border-none text-slate-600 outline-none focus:ring-2 focus:ring-indigo-500">
                        <option value="360">360 Days (1 Year)</option>
                        <option value="180">180 Days (6 Months)</option>
                        <option value="480">480 Days (Long Term)</option>
                        <option value="120">120 Days (Short Term)</option>
                        <option value="84">84 Days (Quick Loan)</option>
                    </select>
                </div>
            </div>

            <div>
                <label class="block text-[10px] font-bold text-slate-400 uppercase mb-2 ml-1 tracking-widest">Credit History Status</label>
                <select name="Credit_History" class="w-full p-4 bg-slate-50 rounded-2xl border-none outline-none focus:ring-2 focus:ring-indigo-500">
                    <option value="1.0">Clean (No defaults / Paid on time)</option>
                    <option value="0.0">Poor (Previous defaults detected)</option>
                </select>
            </div>

            <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div>
                    <label class="block text-[10px] font-bold text-slate-400 uppercase mb-2 ml-1 tracking-widest">Gender</label>
                    <select name="Gender" class="w-full p-4 bg-slate-50 rounded-2xl border-none text-slate-600 outline-none focus:ring-2 focus:ring-indigo-500">
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>
                <div>
                    <label class="block text-[10px] font-bold text-slate-400 uppercase mb-2 ml-1 tracking-widest">Married</label>
                    <select name="Married" class="w-full p-4 bg-slate-50 rounded-2xl border-none text-slate-600 outline-none focus:ring-2 focus:ring-indigo-500">
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                    </select>
                </div>
                <div class="col-span-2 md:col-span-1">
                    <label class="block text-[10px] font-bold text-slate-400 uppercase mb-2 ml-1 tracking-widest">Dependents</label>
                    <select name="Dependents" class="w-full p-4 bg-slate-50 rounded-2xl border-none text-slate-600 outline-none focus:ring-2 focus:ring-indigo-500">
                        <option value="0">0</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3+">3+</option>
                    </select>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label class="block text-[10px] font-bold text-slate-400 uppercase mb-2 ml-1 tracking-widest">Education</label>
                    <select name="Education" class="w-full p-4 bg-slate-50 rounded-2xl border-none text-slate-600 outline-none focus:ring-2 focus:ring-indigo-500">
                        <option value="Graduate">Graduate</option>
                        <option value="Not Graduate">Not Graduate</option>
                    </select>
                </div>
                <div>
                    <label class="block text-[10px] font-bold text-slate-400 uppercase mb-2 ml-1 tracking-widest">Self Employed</label>
                    <select name="Self_Employed" class="w-full p-4 bg-slate-50 rounded-2xl border-none text-slate-600 outline-none focus:ring-2 focus:ring-indigo-500">
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                    </select>
                </div>
            </div>

            <div>
                <label class="block text-[10px] font-bold text-slate-400 uppercase mb-2 ml-1 tracking-widest">Property Area Type</label>
                <select name="Property_Area" class="w-full p-4 bg-slate-50 rounded-2xl border-none text-slate-600 outline-none focus:ring-2 focus:ring-indigo-500">
                    <option value="Semiurban">Semiurban (High Approval Likelihood)</option>
                    <option value="Urban">Urban</option>
                    <option value="Rural">Rural</option>
                </select>
            </div>

            <button type="submit" class="w-full bg-indigo-600 text-white font-bold py-5 rounded-3xl hover:bg-indigo-700 transition-all shadow-xl shadow-indigo-200 transform hover:-translate-y-1 active:scale-[0.98]">
                Analyze Loan Eligibility
            </button>
        </form>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    ApplicantIncome: float = Form(...), LoanAmount: float = Form(...),
    Credit_History: float = Form(...), Married: str = Form(...),
    Gender: str = Form(...), CoapplicantIncome: float = Form(...),
    Loan_Amount_Term: float = Form(...), Dependents: str = Form("0"),
    Education: str = Form("Graduate"), Self_Employed: str = Form("No"),
    Property_Area: str = Form("Semiurban")
):
    data = pd.DataFrame([{
        'Gender': Gender, 'Married': Married, 'Dependents': Dependents,
        'Education': Education, 'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome, 'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount, 'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History, 'Property_Area': Property_Area
    }])

    lr_prob = log_model.predict_proba(data)[0][1]
    dt_prob = tree_model.predict_proba(data)[0][1]

    final_score = (lr_prob + dt_prob) / 2
    status = "APPROVED" if final_score > 0.5 else "REJECTED"


    color = "indigo" if final_score > 0.5 else "rose"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body class="bg-slate-50 min-h-screen p-10 flex flex-col items-center">
        <div class="max-w-4xl w-full grid grid-cols-1 md:grid-cols-2 gap-8">
            
            <div class="bg-white rounded-[40px] shadow-2xl overflow-hidden border border-slate-100 h-fit">
                <div class="bg-{color}-600 p-12 text-center text-white">
                    <h2 class="text-5xl font-black italic">{status}</h2>
<p class="mt-2 opacity-80 uppercase text-xs font-bold tracking-widest">Combined AI Decision</p>                </div>
                <div class="p-8 space-y-4">
                    <div class="p-4 bg-slate-50 rounded-2xl">
                        <p class="text-xs font-bold text-slate-400 uppercase">Input Analysis</p>
                        <p class="text-slate-700 font-bold">Total Confidence: {final_score * 100:.1f}%</p>
                        <p class="text-slate-700 font-bold">Credit: {"Healthy" if Credit_History == 1 else "Poor"}</p>
                        <p class="text-slate-700 font-bold">Income Ratio: {ApplicantIncome/LoanAmount:.2f}x</p>
                    </div>
                    <a href="/" class="block text-center bg-slate-900 text-white py-4 rounded-2xl font-bold">Back</a>
                </div>
            </div>

            <div class="bg-white rounded-[40px] shadow-2xl p-8 border border-slate-100">
                <h3 class="text-lg font-black text-slate-800 mb-6">Model Comparison</h3>
                <canvas id="modelChart" width="400" height="300"></canvas>
                <div class="mt-6 text-xs text-slate-500 italic">
                    <p><b>Logistic Regression:</b> Sees risk as a smooth gradient.</p>
                    <p class="mt-1"><b>Decision Tree:</b> Sees risk as a hard "Pass/Fail" split.</p>
                    <p><b>Final Decision:</b> Calculated by averaging Logistic Regression ({lr_prob*100:.0f}%) and Decision Tree ({dt_prob*100:.0f}%).</p>
                </div>
            </div>

        </div>

        <script>
            const ctx = document.getElementById('modelChart').getContext('2d');
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: ['Logistic Regression', 'Decision Tree', 'Combined Final'],
                    datasets: [{{
                        label: 'Approval Confidence (%)',
                        data: [{lr_prob * 100}, {dt_prob * 100}, {final_score * 100}],
                        backgroundColor: ['#6366f1', '#10b981', '#4f46e5'],
                        borderRadius: 12
                    }}]
                }},
                options: {{
                    scales: {{ y: {{ beginAtZero: true, max: 100 }} }},
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});
        </script>
    </body>
    </html>
    """