import matplotlib
import matplotlib.pyplot as plt
plt.ion()
matplotlib.use('TkAgg')
import numpy as np
from utils_viz import visualization
from utils_gtsam import *
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import argparse
import cv2
from tactile_module import tactile_module

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='rectangle/20220914115144')
args = parser.parse_args()
dataset_name = vars(args)['dataset_name']

data_dir = f'../sample_data/{dataset_name}/'
object_name = dataset_name.split('/')[-2]

cart_init = np.load(data_dir+'cart_init.npy')
cart_seq = np.load(data_dir+'cart_g1_rock.npy').squeeze() # robot pose
grasp_pose = np.load(data_dir+'grasp_pose.npy') # initial grasp pose
comrot_seq = np.load(data_dir+'comrot_g1.npy').squeeze() # desired rotation command
wrench_seq = np.load(data_dir+'wrench.npy') # measured wrench on ground

#### INITIALIZE VISUAL ####

height = grasp_pose[2] + 52
pose_rel = np.zeros(6)
pose_rel[3:] = R.from_quat(cart_init[3:]).as_euler('zyx')
T_wg0 = np.eye(4)
T_wg0[:3,:3] = R.from_quat(cart_init[3:]).as_matrix()
T_wg0[:3,-1] = cart_init[:3]
T_og = np.eye(4)
T_og[:3,:3] = R.from_euler('zyx', grasp_pose[3:]).as_matrix()
T_og[:3,-1] = np.array([0,0,52]) + grasp_pose[:3]    

viz = visualization(54.5, pose_rel, height, T_og, view_elev=40, view_azim=230,\
                    view_center=(0,0,-height), view_radius=35, env_type='floor', object_name=object_name)
viz.set_show(env=False, grp=True, obj_gt=True, obj_est=True, cpoint=True, cline=False)

#########################

tm = tactile_module(TCP_offset=np.array([-6.6, 0, -12.]), r_convert=R.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))
tm.ema_decay_ = 0.
graph = gtsam_graph()
graph.restart(cart_init)

for i in tqdm(range(0,len(cart_seq))):

    # Measured Tactile Images
    g1 = cv2.imread(data_dir+f'g1_{i}.jpg')
    g2 = cv2.imread(data_dir+f'g2_{i}.jpg')

    # Tactile Images with Marker Motion (only for visualization purpose)
    markers_g1 = cv2.imread(data_dir+f'markers_g1_{i}.jpg')
    markers_g2 = cv2.imread(data_dir+f'markers_g2_{i}.jpg')
    cv2.imshow('gelslim', np.vstack((markers_g1,markers_g2)))
    cv2.waitKey(1)

    # Input: Tactile Images
    # Output: Tactile Displacement
    tm.call_back1(g1)
    tm.call_back2(g2)
    tact_disp = np.expand_dims(tm.nn_output, 0)
    
    # Add measurements (robot kinematics, tactile displacement) and desired rotation to the factor graph
    graph.add_new((cart_seq[[i]], tact_disp, comrot_seq[[i]]))

    # Get the results from the factor graph
    ct_est, ct_cov = graph.ct_, graph.ct_cov
    force_est = graph.force_est # world frame
    wrench_est = graph.wr_ # gripper frame
    st = graph.st

    # Turn on the contact formation transition detection accordingly
    if graph.i > 120:
        graph.cf_detect_on_1 = True
    if graph.mode == 1 and graph.i > graph.it + 100:
        graph.cf_detect_on_2 = True
    
    # Visualization

    viz.plot_clear()

    T_wg = np.eye(4)
    T_wg[:3,:3] = R.from_quat(cart_seq[i][3:]).as_matrix()
    T_wg[:3,-1] = cart_seq[i][:3]
    T_gd = np.eye(4)
    T_gd[:3,:3] = R.from_euler('zyx', tact_disp[0,3:], degrees=True).as_matrix()
    T_gd[:3,-1] = tact_disp[0,:3]

    T_g0g = np.linalg.inv(T_wg0) @ T_wg
    T_g0o = np.linalg.inv(T_wg0) @ T_wg @ T_gd @ np.linalg.inv(T_og)

    grp_gt_rot = T_g0g[:3,:3]
    grp_gt_trn = T_g0g[:3,-1]
    obj_gt_rot = T_g0o[:3,:3]
    obj_gt_trn = T_g0o[:3,-1]

    if graph.mode != 0:
        viz.set_show(env=False, grp=True, obj_gt=True, obj_est=True, cpoint=True, cline=True)
        viz.plot_confidence_cone(ct_est.translation(), ct_est.rotation().matrix(), ct_cov[:3,:3])
    else:
        viz.set_show(env=False, grp=True, obj_gt=True, obj_est=True, cpoint=True, cline=False)
    viz.plot_confidence_ellipsoid(ct_est.translation(),
                                    ct_est.rotation().matrix() @ ct_cov[3:,3:] @ ct_est.rotation().matrix().T)
    scale = -45
    scale_ = -8
    ct_trn = viz.pose_rot.dot(ct_est.translation()) + viz.pose_trn
    # force on ground (estimate)
    viz.ax.quiver(ct_trn[0],ct_trn[1],ct_trn[2],scale*force_est[0],scale*force_est[1],scale*force_est[2], color='r', lw=5, arrow_length_ratio=0.1)
    # force on ground (sensor)
    viz.ax.quiver(ct_trn[0],ct_trn[1],ct_trn[2],scale_*wrench_seq[i,0],scale_*wrench_seq[i,1],scale_*wrench_seq[i,2], color='k', lw=5, arrow_length_ratio=0.1)
    coc_trn_ = grp_gt_trn + grp_gt_rot @ st[6:]
    coc_trn = viz.pose_rot.dot(coc_trn_) + viz.pose_trn
    viz.ax.plot3D(coc_trn[0],coc_trn[1],coc_trn[2],'o-',c='r',markersize=20)
    # force on gripper (est)
    viz.ax.quiver(coc_trn[0],coc_trn[1],coc_trn[2],-scale*force_est[0],-scale*force_est[1],-scale*force_est[2], color='r', lw=5, arrow_length_ratio=0.1)
    # torque on gripper
    viz.plot_torque(grp_gt_trn + grp_gt_rot @ st[6:], viz.pose_rot @ grp_gt_rot @ wrench_est[:3], .7, .7, c='r')
    c_wr_vec = - np.cross((coc_trn_ - ct_est.translation()), wrench_est[3:]) - wrench_est[:3]
    # torque on contact (estimate)
    if graph.mode != 0:
        viz.plot_torque(ct_est.translation(), viz.pose_rot @ grp_gt_rot @ c_wr_vec, .7, .7, c='r')

    viz.plot_update(grp_gt_trn, grp_gt_rot,
                    obj_gt_trn, obj_gt_rot,
                    ct_est.translation(), ct_est.rotation().matrix(),
                    obj_gt_trn, obj_gt_rot, alpha=.1)