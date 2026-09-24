"""Resample a ScenarioNet scenario from its logged ~10 Hz steps onto an even grid.

Positions and velocities are linearly interpolated against the true log
timestamps, heading on the circle. An object is valid at t_k only if both
bracketing log samples are valid (one sample when t_k lands on it). Traffic
light states hold their last logged value. The bracketing log timestamps are
kept in metadata["log_ts_lo"/"log_ts_hi"].
"""
import numpy as np

HZ = 7


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def grid(ts, hz=HZ):
    ts = np.asarray(ts, dtype=np.float64) - float(ts[0])
    tk = np.arange(int(np.floor(ts[-1] * hz + 1e-6)) + 1) / hz
    lo = np.clip(np.searchsorted(ts, tk, side="right") - 1, 0, len(ts) - 2)
    hi = lo + 1
    w = np.clip((tk - ts[lo]) / (ts[hi] - ts[lo]), 0.0, 1.0)
    return ts, tk, lo, hi, w


def resample_sd(sd, hz=HZ):
    md = sd["metadata"]
    ts, tk, lo, hi, w = grid(md["ts"], hz)
    K = len(tk)
    on_lo, on_hi = w < 1e-6, w > 1 - 1e-6
    for tr in sd["tracks"].values():
        st = tr["state"]
        v_lo, v_hi = st["valid"][lo].astype(bool), st["valid"][hi].astype(bool)
        valid = np.where(on_lo, v_lo, np.where(on_hi, v_hi, v_lo & v_hi))
        out = {}
        for k, a in st.items():
            a = np.asarray(a)
            if k == "valid":
                out[k] = valid.astype(a.dtype)
            elif k in ("position", "velocity"):
                ww = w.reshape((-1,) + (1,) * (a.ndim - 1))
                out[k] = ((1 - ww) * a[lo] + ww * a[hi]).astype(a.dtype)
            elif k == "heading":
                out[k] = _wrap(a[lo] + w * _wrap(a[hi] - a[lo])).astype(a.dtype)
            else:  # box size and anything else: the valid neighbour's value
                m = v_lo.reshape((-1,) + (1,) * (a.ndim - 1))
                out[k] = np.where(m, a[lo], a[hi]).astype(a.dtype)
            if k != "valid":
                out[k][~valid] = 0
        tr["state"] = out
        tr["metadata"]["track_length"] = K
    held = np.clip(np.searchsorted(ts, tk + 1e-6, side="right") - 1, 0, len(ts) - 1)
    for dm in sd["dynamic_map_states"].values():
        s = dm["state"]["object_state"]
        dm["state"]["object_state"] = [s[i] for i in held]
        if "metadata" in dm:
            dm["metadata"]["track_length"] = K
    ci = int(md.get("current_time_index", 0))
    sd["length"] = K
    md["ts"] = tk.astype(np.float32)
    md["track_length"] = K
    md["current_time_index"] = int(np.searchsorted(tk, ts[min(ci, len(ts) - 1)] - 1e-6))
    md["log_ts_lo"] = ts[lo].astype(np.float32)
    md["log_ts_hi"] = ts[hi].astype(np.float32)
    md["resample_hz"] = hz
    return sd
