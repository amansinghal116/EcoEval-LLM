# ecoeval/core.py
import time
import traceback
from typing import Dict, Any, Optional, List

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.errors import RepositoryNotFoundError

from .config import EcoEvalConfig


# ---------- Prompt template to force clean Python output ----------

PROMPT_TEMPLATE = """
You are an expert Python 3 programmer.

Write ONLY valid Python 3 code.

Requirements:
- Define exactly ONE function that solves the task.
- Do NOT print anything.
- Do NOT include explanations, comments, or examples.
- Do NOT include '>>>' prompts or any natural language text.
- Only return the function definition and any necessary helper code.

Task:
{task}
"""


# ---------- Device + model loading ----------

def _select_device(cfg: EcoEvalConfig) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.device == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(cfg: EcoEvalConfig):
    """
    Load tokenizer and model from Hugging Face Hub.
    Raises a clean RuntimeError if the model id is invalid.
    """
    device = _select_device(cfg)

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
    except (OSError, RepositoryNotFoundError) as e:
        raise RuntimeError(
            f"Could not load model '{cfg.model_id}'. "
            "Make sure it is a valid public model on Hugging Face "
            "(e.g. 'gpt2', 'Salesforce/codegen-350M-mono', "
            "'bigcode/tiny_starcoder_py')."
        ) from e

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.to(device)
    model.eval()
    return tokenizer, model, device


# ---------- Output cleaning / extraction ----------

def _strip_leading_docstring(text: str) -> str:
    """
    Remove a leading triple-quoted docstring if present.
    """
    for quote in ('"""', "'''"):
        if text.startswith(quote):
            parts = text.split(quote)
            if len(parts) >= 3:
                # parts: ["", docstring, rest...]
                return quote.join(parts[2:]).lstrip()
    return text


def _extract_code(generated: str) -> str:
    """
    Clean raw model output into executable Python:

    - Keep from the first 'def ' onwards when possible.
    - Remove triple-quoted docstrings.
    - Drop obvious natural-language lines.
    - Stop at top-level 'if __name__ == "__main__"' or other
      top-level control-flow scaffolding that often causes
      indentation errors.
    """
    text = generated.strip()

    # If there's a function definition, keep from there.
    idx = text.find("def ")
    if idx != -1:
        text = text[idx:]

    # Remove a leading docstring if present.
    text = _strip_leading_docstring(text)

    bad_prefixes = (
        ">>>",
        "Example:",
        "Examples:",
        "Input:",
        "Input Format:",
        "Output:",
        "Output Format:",
        "Python 3:",
        "The function ",
        "The first line ",
        "The above code",
        "The following code",
        "- ",  # bullet lists like "- Write a function ..."
    )

    lines = text.splitlines()
    cleaned: List[str] = []
    in_docstring = False

    for line in lines:
        stripped = line.strip()

        # Track and drop any triple-quoted docstring blocks anywhere
        if '"""' in stripped or "'''" in stripped:
            # toggle docstring state and skip this line
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue

        if not stripped:
            # keep blank lines (can be inside function)
            cleaned.append("")
            continue

        # Drop obvious NL/meta text
        if any(stripped.startswith(bp) for bp in bad_prefixes):
            continue
        if stripped.startswith("```"):
            continue

        # Detect top-level (unindented) scaffolding and stop there
        is_top_level = (line == stripped)  # no leading spaces/tabs

        if is_top_level and stripped.startswith("if __name__"):
            # stop before main-guard
            break

        if is_top_level and stripped.startswith(("for ", "while ", "if ", "elif ", "else:", "try:", "except", "with ")):
            # likely problem-causing scaffold; stop here
            break

        cleaned.append(line)

    code = "\n".join(cleaned).rstrip()
    return code



# ---------- Generation + execution ----------

def generate_code(
    prompt: str,
    tokenizer,
    model,
    cfg: EcoEvalConfig,
    device: torch.device,
) -> str:
    """
    Generate Python code given a full prompt (already templated).
    """
    encoded = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Take the part after the prompt to avoid echoing it.
    if full_text.startswith(prompt):
        raw = full_text[len(prompt):].strip()
    else:
        raw = full_text.strip()

    return _extract_code(raw)


def run_python_tests(pred_code: str, test_code: str) -> bool:
    """
    Very simple sandbox: execs pred_code + test_code in the same namespace.

    NOTE: This is not safe against malicious code. For research/demo only.
    """
    namespace: Dict[str, Any] = {}
    try:
        exec(pred_code, namespace, namespace)
        exec(test_code, namespace, namespace)
        return True
    except Exception:
        traceback.print_exc()
        return False


# ---------- Main benchmark loop ----------

def run_benchmark(
    dataset: Dataset,
    cfg: EcoEvalConfig,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the EcoEval benchmark over a dataset.

    Dataset must have columns:
      - 'prompt'    : natural language description of the task
      - 'test_code' : Python unit tests to validate the solution
    """
    tokenizer, model, device = load_model_and_tokenizer(cfg)

    n = len(dataset)
    if limit is not None:
        n = min(n, limit)

    passed = 0
    total = 0
    per_task: List[Dict[str, Any]] = []

    start = time.time()

    for idx in range(n):
        row = dataset[idx]
        task_text = row["prompt"]
        test_code = row["test_code"]

        # 🔑 ALWAYS wrap the task in our strict code-only template
        full_prompt = PROMPT_TEMPLATE.format(task=task_text)

        t0 = time.time()
        pred_code = generate_code(full_prompt, tokenizer, model, cfg, device)
        ok = run_python_tests(pred_code, test_code)
        t1 = time.time()

        total += 1
        passed += int(ok)

        per_task.append(
            {
                "task_id": idx,
                "prompt_preview": (task_text[:80] + "…") if len(task_text) > 80 else task_text,
                "passed": bool(ok),
                "runtime_s": round(t1 - t0, 3),
            }
        )

    end = time.time()
    elapsed = end - start
    accuracy = passed / total if total > 0 else 0.0

    return {
        "tasks": total,
        "passed": passed,
        "accuracy": accuracy,
        "runtime_seconds": elapsed,
        "per_task": per_task,
    }
