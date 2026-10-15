#!/usr/bin/env python3
"""Independently audit every saved matched trial, retaining partial failures."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    args = parser.parse_args()
    root = args.cohort.resolve()
    definition = json.loads((root / "definition.json").read_text())
    episodes = json.loads((root / "episodes.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    require(summary["definition_sha256"] == sha(root / "definition.json"), "definition changed")
    expected = {(e, c, s) for e in definition["engines"] for c in definition["conditions"] for s in definition["seeds"]}
    observed = [(r["engine"], r["condition"], r["seed"]) for r in episodes]
    require(len(observed) == len(expected) and set(observed) == expected, "cohort coverage mismatch")
    audited = []
    for row in episodes:
        archive = (root / row["archive"]).resolve()
        require(archive.parent == root and sha(archive) == row["archive_sha256"], "archive missing, escaped or changed")
        with np.load(archive, allow_pickle=False) as data:
            states, commands, applied = data["states"], data["commands"], data["applied_actions"]
            rng = np.random.default_rng(row["seed"])
            start = np.r_[rng.uniform(-.08, .08), np.pi + rng.uniform(-.04, .04, 10), rng.uniform(-.02, .02, 11)]
            require(np.array_equal(states[0], start), "seed/reset mismatch")
            require(row["initial_state_sha256"] == hashlib.sha256(start.tobytes()).hexdigest(), "reset hash mismatch")
            complete = states.shape == (1001, 22) and commands.shape == applied.shape == (1000,)
            finite = bool(np.isfinite(states).all() and np.isfinite(commands).all() and np.isfinite(applied).all())
            rail = bool(np.max(np.abs(states[:, 0])) < 2.4)
            force = bool(commands.size and np.max(np.abs(commands)) <= 1.)
            queue = bool(applied.size and applied[0] == 0. and np.array_equal(applied[1:], commands[:-1]))
            hold = False
            if complete:
                angles = (states[750:, 1:11] + np.pi) % (2 * np.pi) - np.pi
                hold = bool(np.max(np.abs(angles)) < .22 and np.max(np.abs(states[750:, 12:])) < 1.)
            gates = {"complete": complete, "finite": finite, "rail": rail, "force": force,
                     "one_tick_queue": queue, "final_251_endpoints": hold}
            if row["condition"] != "classical":
                estimates, residuals = data["inferred_states"], data["inverse_residuals"]
                gates["strict_inverse"] = bool(residuals.shape == (1000,) and np.isfinite(residuals).all() and np.max(residuals) <= 1e-12)
                gates["recovered_state"] = bool(estimates.shape == (1000, 22) and complete
                    and np.max(np.abs(estimates - states[:-1])) < 1e-10)
            gates["pass"] = row["error"] is None and all(gates.values())
            require(gates == row["gates"], f"independent gates disagree: {archive.name}")
            require(row["completed_ticks"] == len(commands) == len(data["physics_ns"]), "tick accounting mismatch")
            require(np.all(data["inference_ns"] >= 0) and np.all(data["physics_ns"] >= 0), "invalid timings")
            audited.append({"engine": row["engine"], "condition": row["condition"], "seed": row["seed"], "pass": gates["pass"]})
    groups = []
    for published in summary["groups"]:
        selected = [r for r in audited if r["engine"] == published["engine"] and r["condition"] == published["condition"]]
        passed = sum(r["pass"] for r in selected)
        require(passed == published["successes"] and len(selected) == published["trials"], "summary count mismatch")
        require([r["seed"] for r in selected if not r["pass"]] == published["failed_seeds"], "failures dropped")
        groups.append({"engine": published["engine"], "condition": published["condition"], "passed": passed, "trials": len(selected)})
    result = {"pass": True, "trials_independently_audited": len(audited), "groups": groups,
              "all_251_final_endpoints_checked": True, "recorded_physics_rerun": False,
              "definition_sha256": sha(root / "definition.json"), "episodes_sha256": sha(root / "episodes.json")}
    (root / "independent-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
