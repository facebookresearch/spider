# Generated data file structure

To handle multiple datasets, robots and tasks, we use the following file structure:

```
example_datasets/
├── raw/
│   └── oakink/
│       ├── some_raw_data_01.pkl
│       └── meshes/
│
└── processed/
    └── oakink/
        ├── dataset_summary.json      # High-level info: lists all tasks, robots, processing date, etc.
        │
        ├── assets/                   # Central pool for all shared assets to avoid duplication
        │   ├── objects/
        │   │   ├── cup/
        │   │   │   ├── convex/
        │   │   │   │   ├── 0.obj
        │   │   │   │   ├── 1.obj
        │   │   │   │   ├── 2.obj
        │   │   │   │   ├── 3.obj
        │   │   │   │   ├── 4.obj
    │   │   │   │   └── visual.obj # original visual mesh for the object
        │   │   ├── jar/
        │   │   │   ├── convex/
        │   │   │   │   ├── 0.obj
        │   │   │   │   ├── 1.obj
        │   │   │   │   ├── 2.obj
        │   │   │   │   ├── 3.obj
        │   │   │   │   ├── 4.obj
        │   │   │   └── visual.obj
        │   │   └── bottle/
        │   │       ├── 0.obj
        │   │       ├── 1.obj
        │   │       └── visual.obj
        │   └── robots/
        │       ├── allegro/
        │       │   ├── left.xml
        │       │   └── right.xml
        │       │   └── bimanual.xml
        │       └── inspire/
        │           ├── left.xml
        │           ├── right.xml
        │           └── bimanual.xml
        ├── allegro/                  # Data processed for the "Allegro" robot
        │   ├── right/                # For the "right" hand configuration
        │   │   ├── lift_cup/         # Task: "lift_cup"
        │   │   │   ├── 01/           # Trial 1 of this task
        │   │   │   │   ├── scene.xml
        │   │   │   │   ├── trajectory_kinematic.npz
        │   │   │   │   ├── trajectory_mjwp.npz # MJX/Warp retargeting
        │   │   │   │   ├── trajectory_isaac.npz # Isaac retargeting
        │   │   │   │   ├── visualization.mp4
        │   │   │   │   └── info.json # Metadata for this trial, including relative paths to assets
        │   │   │   │                 # e.g., "object_asset_paths": {"cup": "../../../../assets/objects/cup/model.obj"}
        │   │   │   │
        │   │   │   └── 02/           # Trial 2 of this task
        │   │   │       ├── scene.xml
        │   │   │       ├── trajectory_kinematic.npz
        │   │   │       ├── trajectory_optimized.npz
        │   │   │       └── info.json
        │   │   │
        │   │   └── pour_water/       # Another task for the same robot/hand setup
        │   │       └── 01/
        │   │           ├── scene.xml
        │   │           ├── trajectory_optimized.npz
        │   │           └── info.json
        │   │
        │   └── bimanual/             # For the "bimanual" (two-hand) configuration
        │       └── open_jar/
        │           └── 01/
        │               ├── scene.xml
        │               ├── trajectory_optimized.npz
        │               └── info.json
        │
        └── mano/              # Extraced raw mano data
            └── bimanual/
                └── right/
                    └── press_button/
                        └── 01/
                        ├── trajectory_keypoint.npz
                        └── info.json
        └── shadow_hand/              # Data processed for the "Shadow Hand" robot
            └── right/
                └── press_button/
                    └── 01/
                        ├── scene.xml
                        ├── trajectory_optimized.npz
                        └── info.json
```
