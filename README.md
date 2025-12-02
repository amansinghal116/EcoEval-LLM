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

**EcoEval-LLM** benchmarks code generation models on:

- ✅ Task correctness (unit-test based pass rate)  
- ⏱ Runtime  
- ⚡ Energy consumption (kWh)  
- 🌍 CO₂ emissions (kg) via [CodeCarbon](https://github.com/mlco2/codecarbon)

It runs a small benchmark of Python programming tasks, executes the generated code against unit tests, and measures the environmental footprint of the run.

## How it works

1. You choose:
   - A Hugging Face Hub model ID (e.g. `Salesforce/codegen-350M-multi`)
   - A built-in Python benchmark dataset
2. The app:
   - Loads the model and tokenizer via `transformers`
   - Generates code for each task
   - Executes unit tests to check correctness
   - Wraps the whole process in a `CodeCarbon.EmissionsTracker` to measure energy and CO₂
3. Results:
   - Run-level summary (accuracy, runtime, energy, CO₂, energy per task, CO₂ per passed task)
   - Per-task pass/fail and runtime
   - Persistent leaderboard (`runs.csv`) across Space sessions

## Run locally

```bash
git clone <this-repo-url>
cd EcoEval-LLM
pip install -r requirements.txt
python app.py
