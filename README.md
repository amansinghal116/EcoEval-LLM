---
title: EcoEval-LLM
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
---

# 🌱 EcoEval-LLM: Energy & Carbon Benchmarking for LLM Code Generation

**EcoEval-LLM** is a lightweight, reproducible framework for evaluating code-generation models across:

- ✅ **Correctness** (unit-test pass rate)  
- ⏱ **Runtime**  
- ⚡ **Energy consumption (kWh)**  
- 🌍 **CO₂ emissions (kg)** via [CodeCarbon](https://github.com/mlco2/codecarbon)

The app runs a small benchmark of Python programming tasks, executes model-generated code, and measures its environmental footprint.

🔗 **Try the live Hugging Face Space:**  
👉 https://huggingface.co/spaces/singhalamaan116/EcoEval-LLM

---

## 🚀 How It Works

1. **You choose:**
   - A Hugging Face model (e.g., `Salesforce/codegen-350M-mono`)
   - A Python benchmark dataset (e.g., `tiny-python-benchmark`)

2. **EcoEval-LLM automatically:**
   - Loads the model using `transformers`
   - Generates code for benchmark prompts
   - Executes and unit-tests the generated solutions
   - Tracks energy + CO₂ using `CodeCarbon.EmissionsTracker`

3. **You get:**
   - **Run-level summary:** accuracy, runtime, energy, CO₂, energy per task, CO₂ per passed task  
   - **Per-task results:** pass/fail and execution latency  
   - **Persistent leaderboard:** stored in `runs.csv` across Space sessions

---

## 🖥 Run Locally

```bash
git clone <this-repo-url>
cd EcoEval-LLM
pip install -r requirements.txt
python app.py
