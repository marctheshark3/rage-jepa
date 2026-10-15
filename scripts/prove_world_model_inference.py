#!/usr/bin/env python3
"""Check the complete WM in an isolated, noneditable NumPy-only wheel install."""
import argparse
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


class NoTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in {"torch", "mujoco", "casadi", "gymnasium"} or fullname in {
                "rage_mill.jepa", "rage_mill.jepa_train", "rage_mill.pole_control", "rage_mill.numpy_policy"}:
            raise AssertionError(f"inference attempted a forbidden training/simulation import: {fullname}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(weights):
    return {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in weights.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="directory containing frozen Torch reference")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    for name in ("torch", "mujoco", "casadi", "gymnasium"):
        if importlib.util.find_spec(name) is not None:
            raise ValueError(f"proof needs an inference-only install; found {name}")
    sys.meta_path.insert(0, NoTrainingImports())
    from rage_mill.world_model import NumpyWorldModel
    import rage_mill.world_model as module
    if not Path(module.__file__).is_relative_to(Path(sys.prefix)):
        raise ValueError("proof must load a noneditable installed wheel")
    root = args.reference
    description = json.loads((root / "reference.json").read_text())
    definition = json.loads((root / "definition.json").read_text())
    if description["definition_sha256"] != sha(root / "definition.json") or description["reference_sha256"] != sha(root / "torch-reference.npz"):
        raise ValueError("reference inputs or declared gates changed")
    models = {dtype: NumpyWorldModel(dtype=dtype) for dtype in ("float32", "float64")}
    before = {dtype: digest(model.weights) for dtype, model in models.items()}
    if sha(Path(module.__file__)) != definition["runtime_code_sha256"]:
        raise ValueError("runtime differs from the reference's declared source")
    checked_values, outcomes = 0, []
    maxima = {dtype: {name: 0. for name in ("online", "target", "predictions")} for dtype in models}
    with np.load(root / "torch-reference.npz", allow_pickle=False) as reference:
        expected_tensors = {name.removeprefix("tensor__"): reference[name]
                            for name in reference.files if name.startswith("tensor__")}
        if set(expected_tensors) != set(models["float32"].weights):
            raise ValueError("full trained tensor set was not exported")
        for model in models.values():
            if model.parameter_count != 107904 or len(model.weights) != 21:
                raise ValueError("full model parameters are missing")
            for name, expected in expected_tensors.items():
                np.testing.assert_array_equal(model.weights[name].astype(np.float32), expected)
        for case in description["cases"]:
            model, name = models[case["dtype"]], case["id"]
            observations, actions = reference[f"{name}_observations"], reference[f"{name}_actions"]
            np.testing.assert_array_equal(model.normalize("npole10", observations), reference[f"{name}_normalized"])
            actual = {"online": model.encode("npole10", observations),
                      "target": model.encode("npole10", observations, target=True),
                      "predictions": model.rollout("npole10", observations, actions)}
            tolerance = definition["acceptance_tolerances"][case["dtype"]]
            for component, values in actual.items():
                expected = reference[f"{name}_{component}"]
                np.testing.assert_allclose(values, expected, **tolerance)
                maxima[case["dtype"]][component] = max(maxima[case["dtype"]][component], float(np.max(np.abs(values - expected))))
                checked_values += values.size
            outcomes.append({**case, "pass": True})
    if any(digest(model.weights) != before[dtype] for dtype, model in models.items()):
        raise ValueError("inference changed model weights")
    blocked = [name for name in sys.modules if name.split(".")[0] in {"torch", "mujoco", "casadi", "gymnasium"}]
    if blocked:
        raise ValueError(f"unexpected training/simulation modules loaded: {blocked}")
    result = {"pass": True, "full_world_model": True, "learned_predictor_executed": True,
              "training": False, "forbidden_import_hook_active": True, "torch_installed": False,
              "torch_imported": False, "simulator_installed": False, "noneditable_wheel": True,
              "parameter_count": 107904, "trained_tensors_verified_exactly": 21,
              "cases_checked": len(outcomes), "output_values_compared": checked_values, "max_absolute_errors": maxima,
              "weights_unchanged": True, "runtime_sha256": sha(Path(module.__file__)),
              "model_sha256": models["float32"].artifact_sha256,
              "reference_sha256": description["reference_sha256"], "cases": outcomes,
              "installed_distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "original_predictive_pass": description["original_predictive_pass"],
              "original_physical_forecast_pass": description["original_physical_forecast_pass"]}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
