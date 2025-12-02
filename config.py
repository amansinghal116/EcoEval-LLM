# ecoeval/config.py
from dataclasses import dataclass


@dataclass
class EcoEvalConfig:
    model_id: str = "Salesforce/codegen-350M-mono"
    max_new_tokens: int = 96      # a bit shorter to reduce rambling
    temperature: float = 0.0      # deterministic, less chatty
    top_p: float = 0.95
    device: str = "auto"

