import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import gtsam
import matplotlib.pyplot as plt
plt.ion()
from utils_viz import visualization
from utils_gtsam import *
from utils_simulation import graph_simulator
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

viz_on = True

MU = 0.2
height = 62
grasp_pose = np.array([0,0,0,0,0,0])
pose_rel = np.zeros(6)
cart_init = np.array([0,0,0,0,0,0,1])
contact_point = np.array([-17.5,-25,0]) # position of the contact point in the object bottom center frame
touch_begin = 0

traj = np.zeros((270,7))
traj[:,-1] = 1
traj[:20,2] = np.linspace(0,-.2,20)
traj[20:,2] = -.2
traj[20:70,1] = np.linspace(0,2,50)
traj[70:270,0] = 2*np.sin(np.linspace(0,2*np.pi,200))
traj[70:270,1] = 2*np.cos(np.linspace(0,2*np.pi,200))

simulator = graph_simulator(MU=MU)

simulator.restart(cart_init, grasp_pose, height, contact_point)

if viz_on:
    viz = visualization(height, pose_rel, height, simulator.T_og, view_elev=30, view_azim=225, view_center=(0,0,-height), view_radius=30, env_type='floor')
    viz.set_show(env=False, grp=True, obj_gt=True, obj_est=True, cline=False)

force_log, grp_log, ct_log, tact_log = [], [], [], []

for tra in tqdm(traj):

    simulator.add_new(tra)
    force_log.append(simulator.force)
    grp_log.append(simulator.grp.translation())
    ct_log.append(simulator.ct.translation())
    tact_log.append(np.hstack((simulator.gg_.translation(), R.from_matrix(simulator.gg_.rotation().matrix()).as_euler('zyx',True))))

    if simulator.i > touch_begin:
        simulator.touch = True

    if viz_on:
        viz.plot_clear()
        viz.plot_coordinate_axis(simulator.obj.translation(), simulator.obj.rotation().matrix(), lw=4, c='r')
        viz.plot_coordinate_axis(simulator.nob.translation(), simulator.nob.rotation().matrix(), c='b')
        viz.plot_update(simulator.grp.translation(), simulator.grp.rotation().matrix(),
                simulator.obj.translation(), simulator.obj.rotation().matrix(),
                simulator.ct.translation(), simulator.ct.rotation().matrix(),
                simulator.nob.translation(), simulator.nob.rotation().matrix(), alpha=.1)

force_log, grp_log, ct_log = np.array(force_log), np.array(grp_log), np.array(ct_log)

plt.figure()
plt.plot(force_log)
plt.legend(['x','y','z'])

plt.figure()
plt.plot(-np.linalg.norm(force_log[:,:2],axis=1)/(force_log[:,2]-1e-2))

plt.figure()
plt.plot(ct_log)

plt.figure()
plt.plot(grp_log)

plt.show()

input()