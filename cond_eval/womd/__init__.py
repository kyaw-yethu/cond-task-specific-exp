"""WOMD training_20s -> womd_7hz_f33 (DATASET_PLAN.md in /root/waymo-rendered-dataset).

convert.py   tfrecord shard -> ScenarioNet scenarios resampled to 7 Hz + per-window log stats
select.py    train/val window pairs and the sliced test set
render.py    MetaDrive rollout of a selected scenario -> two 33-frame clips with labels
"""
