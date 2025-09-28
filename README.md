## Overview

Tested on Ubuntu 22.04, isaac sim 5.0. Should also work with isaac sim 4.0/4.5.

Requirement: Basic Isaac Lab, please follow the offcial document to install the environment and for trouble shooting: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

What has been implemented:

1. Load adam_u robot from urdf to Isaac lab.
2. Simple RL environment training grasping using RSL_RL.
3. Arg to choose from different environment.

Notice: This is an example of loading adam_u robot and rl-training, the algorithm and environment design needs to be redesigned.

## Scripts

To try the simple RL environment, you can run navigate to the 'manipulation' folder, and run the command:

```
### Train
python adam_u_rl/scripts/train.py --headless

### Evaluate
python adam_u_rl/scripts/play.py --checkpoint_path logs/rsl_rl/adam_u_grasp/2025-06-27_17-04-25/model_1499.pt
```
