# ecoeval/energy.py
import os
from typing import Callable, Dict, Any

from codecarbon import EmissionsTracker


def run_with_energy(
    fn: Callable[[], Dict[str, Any]],
    project_name: str = "EcoEval-LLM",
) -> Dict[str, Any]:
    """
    Wrap any benchmark function with CodeCarbon energy & emissions tracking.
    """

    # 🔧 Ensure the output directory exists (this fixes your OSError)
    output_dir = "emissions"
    os.makedirs(output_dir, exist_ok=True)

    tracker = EmissionsTracker(
        project_name=project_name,
        measure_power_secs=1,
        output_dir=output_dir,
        save_to_file=True,
        log_level="error",
    )

    tracker.start()
    metrics = fn()
    emissions = tracker.stop()

    # Depending on CodeCarbon version, tracker.stop() may be:
    #   - a float (kg CO2eq)
    #   - a dict with 'energy_kwh' and 'emissions_kg'
    energy_kwh = None
    emissions_kg = None

    if isinstance(emissions, dict):
        energy_kwh = emissions.get("energy_kwh", None)
        emissions_kg = emissions.get("emissions_kg", None)
    else:
        # old style: just kg CO2eq
        emissions_kg = emissions

    metrics["energy_kwh"] = float(energy_kwh) if energy_kwh is not None else None
    metrics["emissions_kg"] = float(emissions_kg) if emissions_kg is not None else None

    return metrics
