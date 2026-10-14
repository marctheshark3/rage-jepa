"""Episode-preserving pole data; public extraction of the frozen collector."""
from __future__ import annotations
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

ENVIRONMENTS = ("npole10",)
ALL_ENVIRONMENTS = tuple(f"npole{n}" for n in range(1, 11))
ENGINE_SUPPORT = {env: ("euler", "rk4", "mujoco") for env in ALL_ENVIRONMENTS}
PREFERRED_ENGINES = {env: ("euler", "mujoco") for env in ALL_ENVIRONMENTS}


def supported_engines(env_id: str) -> tuple[str, ...]:
    if env_id not in ENGINE_SUPPORT:
        raise ValueError(f"Unknown environment {env_id!r}; choose from {ALL_ENVIRONMENTS}")
    return ENGINE_SUPPORT[env_id]


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass
class Trajectories:
    """Fixed-length, complete episodes; no transitions across episode resets.

    Hardware imports require measured timestamps. The rate in metadata alone
    cannot establish a 50 Hz recording, and neither kind of dataset establishes
    identified actuator dynamics.
    """

    observations: np.ndarray
    actions: np.ndarray
    metadata: dict[str, Any]
    timestamps: np.ndarray | None = None

    @property
    def episodes(self) -> int:
        return self.observations.shape[0]

    @property
    def steps(self) -> int:
        return self.actions.shape[1]

    @property
    def observation_dim(self) -> int:
        return self.observations.shape[2]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[2]

    def validate(self) -> Trajectories:
        obs, acts = np.asarray(self.observations), np.asarray(self.actions)
        if obs.ndim != 3 or acts.ndim != 3:
            raise ValueError("observations and actions must have [episode, time, feature] dimensions")
        if min(obs.shape) < 1 or min(acts.shape) < 1:
            raise ValueError("trajectory dimensions must be nonempty")
        if obs.shape[0] != acts.shape[0] or obs.shape[1] != acts.shape[1] + 1:
            raise ValueError("each episode needs exactly one more observation than actions")
        if obs.dtype.kind not in "fiu" or acts.dtype.kind not in "fiu":
            raise ValueError("observations and actions must contain real numeric values")
        if not np.isfinite(obs).all() or not np.isfinite(acts).all():
            raise ValueError("trajectories must contain finite values")
        if np.any(np.abs(acts) > 1.0):
            raise ValueError("actions must be normalized to [-1, 1]")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        required = ("env_id", "body", "source_kind", "engine", "control_hz", "q_semantics", "source_hashes")
        missing = [key for key in required if key not in self.metadata]
        if missing:
            raise ValueError(f"metadata missing provenance: {', '.join(missing)}")
        for key in ("env_id", "body", "engine", "q_semantics"):
            if not isinstance(self.metadata[key], str) or not self.metadata[key].strip():
                raise ValueError(f"metadata {key} must be a nonempty string")
        kind = self.metadata["source_kind"]
        if kind not in ("simulation", "hardware"):
            raise ValueError("source_kind must explicitly be simulation or hardware")
        hz = self.metadata["control_hz"]
        if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
            raise ValueError("control_hz must be a positive finite number")
        hashes = self.metadata["source_hashes"]
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("source_hashes must identify at least one source")
        if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
               for k, v in hashes.items()):
            raise ValueError("source_hashes must map source names to lowercase SHA-256 digests")
        if self.metadata.get("identified", False) is not False:
            raise ValueError("trajectory data cannot establish identified actuator dynamics")
        if self.metadata.get("sim2real", False) is not False:
            raise ValueError("trajectory data cannot establish hardware transfer")
        labels = self.metadata.get("observation_keys")
        if labels is not None and (not isinstance(labels, list) or len(labels) != obs.shape[2]):
            raise ValueError("observation_keys must match the observation dimension")
        if kind == "hardware" and self.timestamps is None:
            raise ValueError("hardware trajectories require actual timestamps, not only a declared control_hz")
        if kind == "hardware" and (
            self.metadata["engine"] in ("euler", "rk4", "mujoco", "numpy", "simulation")
            or self.metadata["q_semantics"].startswith("simulated_")
            or self.metadata.get("model_kind")
        ):
            raise ValueError("simulation provenance cannot be relabelled as hardware")
        if self.timestamps is not None:
            t = np.asarray(self.timestamps)
            if t.shape != obs.shape[:2] or t.dtype.kind not in "fiu" or not np.isfinite(t).all():
                raise ValueError("timestamps must be finite [episode, observation time] values")
            intervals = np.diff(t.astype(np.float64), axis=1)
            if np.any(intervals <= 0):
                raise ValueError("timestamps must increase strictly within each episode")
            if kind == "hardware":
                self.metadata["measured_control_hz"] = float(1.0 / np.median(intervals))
                self.metadata["sampling_matches_declared_hz"] = bool(
                    np.mean(np.abs(intervals - 1.0 / hz) <= 0.05 / hz) >= 0.95
                )
                # q / Vin / delay and system identification are separate evidence.
                self.metadata["hardware_log_gate_passed"] = False
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain finite JSON values") from exc
        self.observations, self.actions = obs, acts
        return self

    def save(self, path: Path | str) -> None:
        save_trajectories(self, path)

    @classmethod
    def load(cls, path: Path | str) -> Trajectories:
        return load_trajectories(path)


def save_trajectories(data: Trajectories, path: Path | str) -> None:
    """Save numeric arrays and a JSON scalar, without pickled objects."""
    data.validate()
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "observations": data.observations,
        "actions": data.actions,
        "metadata": np.array(json.dumps(data.metadata, sort_keys=True, allow_nan=False)),
    }
    if data.timestamps is not None:
        arrays["timestamps"] = np.asarray(data.timestamps)
    with dest.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_trajectories(path: Path | str) -> Trajectories:
    """Import an explicitly labelled recording; hardware needs timestamps."""
    with np.load(path, allow_pickle=False) as archive:
        missing = {"observations", "actions", "metadata"} - set(archive.files)
        if missing:
            raise ValueError(f"trajectory archive missing {sorted(missing)}")
        encoded = archive["metadata"]
        if encoded.ndim != 0 or encoded.dtype.kind not in "US":
            raise ValueError("archive metadata must be a JSON string scalar")
        try:
            scalar = encoded.item()
            metadata = json.loads(scalar.decode("utf-8") if isinstance(scalar, bytes) else scalar)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("archive metadata is not valid JSON") from exc
        result = Trajectories(
            archive["observations"].copy(), archive["actions"].copy(), metadata,
            archive["timestamps"].copy() if "timestamps" in archive.files else None,
        )
    return result.validate()


def sample_windows(
    data: Trajectories, horizon: int, batch_size: int, rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample [B,H+1,O] and [B,H,A] windows wholly inside single episodes."""
    data.validate()
    _positive_integer(horizon, "horizon")
    _positive_integer(batch_size, "batch_size")
    if horizon > data.steps:
        raise ValueError("horizon exceeds the episode length")
    rng = rng if rng is not None else np.random.default_rng()
    episodes = rng.integers(data.episodes, size=batch_size)
    starts = rng.integers(data.steps - horizon + 1, size=batch_size)
    return (
        data.observations[episodes[:, None], starts[:, None] + np.arange(horizon + 1)],
        data.actions[episodes[:, None], starts[:, None] + np.arange(horizon)],
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _metadata(env_id: str, engine: str, seed: int, paths: list[Path]) -> dict[str, Any]:
    return {
        "schema": "rage_mill.jepa.trajectories.v1",
        "env_id": env_id,
        "body": env_id,
        "source_kind": "simulation",
        "engine": engine,
        "control_hz": 50.0,
        "seed": int(seed),
        "identified": False,
        "sim2real": False,
        "hardware_log_gate_passed": False,
        "source_hashes": {str(path.resolve()): _hash(path) for path in paths},
    }


def _actions(rng: np.random.Generator, episodes: int, steps: int, dim: int) -> np.ndarray:
    raw = rng.uniform(-1.0, 1.0, (episodes, steps, dim))
    actions = np.zeros_like(raw, dtype=np.float32)
    state = rng.uniform(-0.25, 0.25, (episodes, dim))
    for t in range(steps):
        state = 0.75 * state + 0.25 * raw[:, t]
        actions[:, t] = state
    return actions


def _npole_initial(rng: np.random.Generator, episodes: int, n: int) -> np.ndarray:
    q = rng.uniform(-0.12, 0.12, (episodes, n + 1))
    q[:, 1:] += np.where(np.arange(episodes)[:, None] % 2 == 0, math.pi, 0.0)
    return np.concatenate([q, rng.uniform(-0.04, 0.04, q.shape)], axis=1)


def _npole_obs(state: np.ndarray, action: np.ndarray, n: int) -> np.ndarray:
    d = n + 1
    q, vel = state[:, :d], state[:, d:]
    parts = [q[:, 0], vel[:, 0]]
    for i in range(n):
        parts += [np.sin(q[:, i + 1]), np.cos(q[:, i + 1]), vel[:, i + 1] / 8.0]
    return np.stack(parts + [action[:, 0]], axis=-1).astype(np.float32)


def _npole_xml(n: int) -> str:
    """Independent MuJoCo mechanics, using the existing mill's geometry.

    Plant1 uses a uniform rod with its COM 0.5 m from the hinge. Chain n>=2
    uses point masses at 0.5 m link tips and absolute-angle coordinates.
    """
    length = 1.0 if n == 1 else 0.5
    bodies, closes = [], []
    for i in range(n):
        inertial = ('<inertial pos="0 0 0.5" mass="0.1" diaginertia="0.008333333333 0.008333333333 0.000001"/>'
                    if n == 1 else '<inertial pos="0 0 0.5" mass="0.1" diaginertia="0.000001 0.000001 0.000001"/>')
        bodies.append(f'<body name="p{i+1}" pos="0 0 {0 if i == 0 else length}">'
                      f'<joint name="h{i+1}" type="hinge" axis="0 1 0"/>'
                      f'<geom type="capsule" fromto="0 0 0 0 0 {length}" size="0.02" mass="0" contype="0" conaffinity="0"/>{inertial}')
        closes.append('</body>')
    return ('<mujoco model="jepa-npole"><option gravity="0 0 -9.81" timestep="0.005" integrator="RK4"/>'
            '<worldbody><body name="cart"><joint name="slide" type="slide" axis="1 0 0"/>'
            '<inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>'
            + ''.join(bodies) + ''.join(closes) + '</body></worldbody>'
            '<actuator><motor name="u" joint="slide" gear="10" ctrllimited="true" ctrlrange="-1 1"/></actuator></mujoco>')


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo collector requires `uv sync --extra jepa --extra mujoco`") from exc
    return mujoco


def _checked_mj_step(mj, model, data) -> None:
    previous_time = float(data.time)
    mj.mj_step(model, data)
    for name in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC"):
        if data.warning[int(getattr(mj.mjtWarning, name))].number:
            raise RuntimeError("MuJoCo reported invalid dynamics; refusing a silently reset trajectory")
    if not np.isclose(data.time, previous_time + model.opt.timestep, rtol=1e-7, atol=1e-10):
        raise RuntimeError("MuJoCo time reset or failed to advance; refusing the trajectory")


def _collect_npole(env_id: str, episodes: int, steps: int, seed: int, engine: str,
                   physics_substeps: int | None = None) -> Trajectories:
    n = int(env_id.removeprefix("npole"))
    substeps = physics_substeps if physics_substeps is not None else (20 if n >= 4 else 4)
    _positive_integer(substeps, "physics_substeps")
    physics_dt = 0.02 / substeps
    # The historical mill helper internally subdivides Euler again for n>=6.
    # Keep old default evolution, but an explicit collector accuracy override
    # and new long chains each use exactly the requested physical step count.
    direct_euler = engine == "euler" and n > 1 and (physics_substeps is not None or n >= 8)
    internal_euler_substeps = (16 if n >= 9 else 8 if n >= 8 else 4 if n >= 6 else 1)
    internal_euler_substeps = internal_euler_substeps if engine == "euler" and not direct_euler else 1
    actual_substeps = substeps * internal_euler_substeps
    rng = np.random.default_rng(seed)
    initial = _npole_initial(rng, episodes, n)
    actions = _actions(rng, episodes, steps, 1)
    observations = np.empty((episodes, steps + 1, 3 + 3 * n), dtype=np.float32)
    observations[:, 0] = _npole_obs(initial, np.zeros((episodes, 1)), n)
    source = Path(__file__).with_name("npole.py")
    metadata = _metadata(env_id, engine, seed, [source, Path(__file__)])
    metadata.update({
        "q_semantics": "simulated_absolute_angle_rad",
        "actuator_model": "normalized_cart_force_10N",
        "physics_device": "cpu", "physics_hz": 50.0 * actual_substeps, "delay_steps": 0,
        "physics_substeps": actual_substeps,
        "requested_physics_substeps": substeps,
        "legacy_internal_euler_substeps": internal_euler_substeps,
        "physics_dt_s": .02 / actual_substeps,
        "initial_distribution": "alternating hanging and upright, angles ±0.12 rad, velocities ±0.04",
        "episode_resets": "only between episodes",
        "observation_keys": ["cart_x_m", "cart_velocity_m_s"]
            + [key for i in range(n) for key in (f"sin_theta_{i+1}", f"cos_theta_{i+1}", f"omega_{i+1}_rad_s_div8")]
            + ["last_cart_force_normalized"],
    })
    if engine == "mujoco":
        mj = _mujoco()
        xml = _npole_xml(n)
        model = mj.MjModel.from_xml_string(xml)
        model.opt.timestep = physics_dt
        metadata["source_hashes"]["generated_mjcf"] = hashlib.sha256(xml.encode()).hexdigest()
        metadata["mujoco_version"] = mj.__version__
        for ep in range(episodes):
            data = mj.MjData(model)
            q, v = initial[ep, :n + 1], initial[ep, n + 1:]
            data.qpos[0], data.qvel[0] = q[0], v[0]
            data.qpos[1:] = np.diff(np.r_[0.0, q[1:]])
            data.qvel[1:] = np.diff(np.r_[0.0, v[1:]])
            mj.mj_forward(model, data)
            for t in range(steps):
                data.ctrl[:] = actions[ep, t]
                for _ in range(substeps):
                    # Convert the mill's absolute-coordinate viscous friction
                    # to MuJoCo's relative hinge generalized forces.
                    data.qfrc_applied[:] = 0.0
                    data.qfrc_applied[0] = -0.05 * data.qvel[0]
                    if n > 1:
                        absolute_vel = np.cumsum(data.qvel[1:])
                        data.qfrc_applied[1:] = np.cumsum((-0.05 * absolute_vel)[::-1])[::-1]
                    _checked_mj_step(mj, model, data)
                state = np.r_[data.qpos[0], np.cumsum(data.qpos[1:]),
                              data.qvel[0], np.cumsum(data.qvel[1:])][None]
                observations[ep, t + 1] = _npole_obs(state, actions[ep:ep+1, t], n)[0]
    else:
        import torch
        from . import npole

        state = torch.tensor(initial, dtype=torch.float64, device="cpu")
        if n == 1:
            state = state[:, [0, 2, 1, 3]]
            plant = npole.Plant1()
            fn = npole.step_euler1 if engine == "euler" else npole.step_rk41
            for t in range(steps):
                a = torch.tensor(actions[:, t, 0], dtype=state.dtype)
                for _ in range(substeps):
                    state = fn(state, a, plant, dt=physics_dt)
                # The n=1 mill state interleaves positions and velocities.
                ordered = state[:, [0, 2, 1, 3]].numpy()
                observations[:, t+1] = _npole_obs(ordered, actions[:, t], n)
        else:
            chain = npole.Chain(n)
            m = torch.full((n,), chain.m, dtype=state.dtype)
            length = torch.full((n,), chain.length, dtype=state.dtype)
            fn = npole.step_euler_n if engine == "euler" else npole.step_rk4_n
            for t in range(steps):
                a = torch.tensor(actions[:, t, 0], dtype=state.dtype)
                for _ in range(substeps):
                    if direct_euler:
                        dim = n + 1
                        q, velocity = state[:, :dim], state[:, dim:]
                        acceleration = npole.qdd_n(q, velocity, a, chain, m, length, None)
                        velocity = velocity + physics_dt * acceleration
                        state = torch.cat((q + physics_dt * velocity, velocity), dim=1)
                    else:
                        state = fn(state, a, chain, m, length, None, dt=physics_dt)
                observations[:, t+1] = _npole_obs(state.numpy(), actions[:, t], n)
    return Trajectories(observations, actions, metadata).validate()


def collect_trajectories(
    env_id: str, episodes: int = 32, steps: int = 64, seed: int = 0,
    engine: str = "euler", *, physics_substeps: int | None = None,
) -> Trajectories:
    """Collect complete pole episodes on CPU with explicit integration accuracy."""
    _positive_integer(episodes, "episodes")
    _positive_integer(steps, "steps")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if engine not in supported_engines(env_id):
        raise ValueError(f"Unsupported engine {engine!r} for {env_id}")
    return _collect_npole(env_id, episodes, steps, seed, engine, physics_substeps)
