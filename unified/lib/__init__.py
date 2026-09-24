"""nuScenes and Waymo in one clip format at the stage-2 training geometry.

`spec` fixes the geometry and the label maths, `sources` reduces each dataset's
own tables to a common per-scene record, `build` cuts one scene into clips, and
`volume` moves bytes to and from the VESSL volume.
"""
