# Tactile-Estimator-Controller

This repository contains the source code of the paper "Simultaneous Tactile Estimation and Control of Extrinsic Contact". You can run the experiment on a real robot or just simply run a replay of the sample dataset.

## Setup Manual

- Clone the repository and create a virtual environment.

  ```
  cd <project foler>
  git clone --recursive git@github.com:sangwkim/Tactile-Estimator-Controller.git
  cd Tactile-Estimator-Controller
  conda env create -f conda_env.yml
  ```
  
- Install prerequisites for the GTSAM library

  ```
  sudo apt-get install libboost-all-dev
  sudo apt-get install cmake
  sudo ldconfig
  ```
  
  If you face a cmake version related error later, you might have to remove the current cmake and reinstall a new version of cmake (!be careful to do this if you are using ROS!, since it can break your ROS setup. In that case, please refer to: https://answers.ros.org/question/293119/how-can-i-updateremove-cmake-without-partially-deleting-my-ros-distribution/)
  
  ```
  sudo apt remove cmake
  sudo apt autoremove
  (Download a new version of cmake eg. 3.22.2)
  cd Downloads/
  sudo cp cmake-3.22.2-linux-x86_64.sh /opt/
  chmod +x /opt/cmake-3.22.2-linux-x86_64.sh 
  sudo bash /opt/cmake-3.22.2-linux-x86_64.sh
  ```
  
- Install GTSAM

  ```
  cd <project folder>
  git clone https://github.com/borglab/gtsam.git
  git checkout 69a3a75
  cd gtsam
  pip install -r python/requirements.txt 
  mkdir build
  cd build
  cmake .. -DGTSAM_PYTHON_VERSION=3.8 -DGTSAM_BUILD_PYTHON=1 -DCMAKE_INSTALL_PREFIX="./install"
  make
  make install
  make python-install
  ```
  
- Install custom factors

  ```
  cd <project folder>/Tactile-Estimator-Controller/custom-factors
  mkdir build
  cd build
  cmake .. -DCMAKE_PREFIX_PATH="<project folder>/gtsam/build/install"
  make
  make python-install
  ```

## Further details

For brief explanation about each custom factor, please refer to the [FACTORS.md](FACTORS.md)

## Directories/Codes description

- custom-factors/:
  - contains a library of custom factors that are used in the paper
- sample_data/:
  - contains a sample dataset for the replay
- weights/:
  - contains a neural network that is used for the tactile module
- codes/tactile_module.py:
  - contains a class for the tactile sensors
- codes/utils_gtsam.py:
  - contains a class for the factor graph models
- codes/replay.py:
  - replay the sample data that were collected in the real experiments
- codes/utils_viz.py:
  - contains a class for visualization
- codes/run.py:
  - main file that runs the real experiment. You need to setup hardware to run this. Otherwise, if just running the replay, this file is not necessary.
- codes/packing_environment.py:
  - contains classes and functions that are relevant to robot configurations and control. This is also not necessary if you are not running this on real hardware

## Manual for running a real experiments

1. roscore
2. run robot node (eg. roslaunch abb.launch)
3. run gripper node (eg. roslaunch wsg_50_driver wsg_50_tcp.launch)
4. run GelSlim nodes (eg. roslaunch raspberry.launch)
5. run factor graph node (eg. python3 utils_gtsam_optimal.py)
6. run main node (eg. python3 main_optimal.py)
