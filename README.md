# active_extrinsic

## Codes description

- codes_real_experiments/main_optimal.py:
  - main file that is used when doing the real experiments
- codes_real_experiments/packing_environment_optimal.py:
  - contains classes and functions that are relevant to robot configurations and control
- codes_real_experiments/tactile_module_optimal.py:
  - contains a class for the tactile sensors
- codes_real_experiments/utils_gtsam_optimal.py:
  - contains a class for the factor graph models
- codes_real_experiments/FG_simulation.py:
  - runs an example for the factor graph simulator
- codes_real_experiments/replay_data.py:
  - replay the data that were collected in the real experiments
- codes_real_experiments/utils_viz.py:
  - contains a class for visualization

## Manual for running a real experiments

1. roscore
2. run robot node (eg. roslaunch abb.launch)
3. run gripper node (eg. roslaunch wsg_50_driver wsg_50_tcp.launch)
4. run GelSlim nodes (eg. roslaunch raspberry.launch)
5. run factor graph node (eg. python3 utils_gtsam_optimal.py)
6. run main node (eg. python3 main_optimal.py)
