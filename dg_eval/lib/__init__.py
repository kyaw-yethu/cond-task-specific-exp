"""DrivingGen's evaluation suite, run on nuScenes.

The library half of the stage-2 evaluation. `nusc_index` turns the raw nuScenes
tables into a compact per-scene index, `clips` cuts calibrated clips from the
CAM_FRONT shards, `extract` is DrivingGen's ego-trajectory extractor with the
patches its released code needs to run off their own data, and `score` applies
their alignment and metrics.
"""
