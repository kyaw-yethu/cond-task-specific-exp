"""Per-window statistics from the resampled log, used to pick windows and test slices.

Computed from the same resampled states MetaDrive replays, so they agree with
the rendered labels. Frame f of a rendered rollout shows resampled step
f + FRAME_OFFSET (MetaDrive records the first frame after one env.step).
"""
import numpy as np

CLIP = 33
STARTS = (0, 33, 66, 99)
PAIRS = ((0, 66), (33, 99))
FRAME_OFFSET = 1

STATIC_M = 1.0          # moves less than this over the window -> static
TURN_DEG = 30.0
TTC_S = 4.0
BRAKE_MPS = 3.0
VRU_M = 20.0
CONE_DEG = 30.0
LEAD_MAX_M = 60.0
LEAD_HALF_WIDTH_M = 2.0

SLICES = ("turn", "lead_closing", "braking", "vru_nearby", "cruising", "static")
SLICE_CODE = {s: i for i, s in enumerate(SLICES)}


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def window_stats(sd, start):
    """Stats over frames [start, start+CLIP) of the rollout, or None if the log is too short."""
    a = start + FRAME_OFFSET
    b = a + CLIP
    if b > sd["length"]:
        return None
    ego = sd["tracks"][sd["metadata"]["sdc_id"]]["state"]
    if not ego["valid"][a:b].all():
        return None
    p = ego["position"][a:b, :2].astype(np.float64)
    h = ego["heading"][a:b].astype(np.float64)
    v = ego["velocity"][a:b].astype(np.float64)
    speed = np.hypot(v[:, 0], v[:, 1])
    path = float(np.hypot(*np.diff(p, axis=0).T).sum())
    dh = float(np.degrees(np.sum(_wrap(np.diff(h)))))
    brake = float(np.max(np.maximum.accumulate(speed) - speed))

    min_ttc, vru = np.inf, False
    c, s = np.cos(h), np.sin(h)
    cone = np.radians(CONE_DEG)
    lead_fwd = np.full(CLIP, np.inf)
    lead_ttc = np.full(CLIP, np.inf)
    for tid, tr in sd["tracks"].items():
        if tid == sd["metadata"]["sdc_id"]:
            continue
        st = tr["state"]
        ok = st["valid"][a:b].astype(bool)
        if not ok.any():
            continue
        d = st["position"][a:b, :2] - p
        fwd = d[:, 0] * c + d[:, 1] * s
        lat = -d[:, 0] * s + d[:, 1] * c
        dist = np.hypot(fwd, lat)
        in_cone = ok & (fwd > 0) & (np.abs(np.arctan2(lat, fwd)) < cone)
        if tr["type"] in ("PEDESTRIAN", "CYCLIST") and (in_cone & (dist < VRU_M)).any():
            vru = True
        is_lead = ok & (fwd > 0) & (np.abs(lat) < LEAD_HALF_WIDTH_M) & (dist < LEAD_MAX_M)
        if not is_lead.any():
            continue
        rv = st["velocity"][a:b] - v
        closing = -(d[:, 0] * rv[:, 0] + d[:, 1] * rv[:, 1]) / (dist + 1e-6)
        nearer = is_lead & (fwd < lead_fwd)
        lead_fwd = np.where(nearer, fwd, lead_fwd)
        ttc = np.where(closing > 0.1, dist / np.maximum(closing, 1e-6), np.inf)
        lead_ttc = np.where(nearer, ttc, lead_ttc)
    min_ttc = float(lead_ttc.min())

    return dict(start=start, path_m=path, dheading_deg=dh, brake_mps=brake,
                min_ttc_s=min_ttc, vru=vru, static=path < STATIC_M)


def slice_of(ws):
    if ws["static"]:
        return "static"
    if abs(ws["dheading_deg"]) > TURN_DEG:
        return "turn"
    if ws["min_ttc_s"] < TTC_S:
        return "lead_closing"
    if ws["brake_mps"] > BRAKE_MPS:
        return "braking"
    if ws["vru"]:
        return "vru_nearby"
    return "cruising"
