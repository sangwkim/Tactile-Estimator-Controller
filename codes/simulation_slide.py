import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from utils_viz import visualization
from utils_gtsam import *
from utils_simulation import graph_simulator
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from tqdm import tqdm

cart_init = np.array([6.60219357e+01, -2.09996375e+02,  7.54190527e+01,  1.25534386e-04, 1.42766641e-01,  4.42155836e-02,  9.88768250e-01])
grasp_pose = np.array([0, 0, 0, 0, 0, -0.3])

show_viz = True
MU = 0.2
height = 52
pose_rel = np.zeros(6)
pose_rel[3:] = R.from_quat(cart_init[3:]).as_euler('zyx')
contact_point = np.array([17.5,-25,0])
touch_begin = 5
coc_offset = np.array([0,0,-7.5])

def tcp_coc(cart_tcp, offset):
    cart_coc = cart_tcp.copy()
    cart_coc[:3] += R.from_quat(cart_tcp[3:]).apply(offset)
    return cart_coc

def coc_tcp(cart_coc, offset):
    cart_tcp = cart_coc.copy()
    cart_tcp[:3] -= R.from_quat(cart_coc[3:]).apply(offset)
    return cart_tcp

grasp_pose[:3] += R.from_euler('zyx', grasp_pose[3:]).apply(coc_offset)

cart_init_offset = coc_tcp(cart_init, coc_offset)

graph = gtsam_graph()
graph.restart(cart_init_offset)

simulator = graph_simulator(MU = MU,
                            OU_sigma = np.array([1e-4, 1e-4, 1e-4, 5e-3, 5e-3, 5e-3]),
                            OU_theta = 0.01)
simulator.restart(cart_init, grasp_pose, height, contact_point)

if show_viz:
    viz = visualization(height, pose_rel, height, simulator.T_og, view_elev=40, view_azim=230, view_center=(0,0,-height), view_radius=30, env_type='floor')
    viz.set_show(env=False, grp=True, obj_gt=True, obj_est=True, cpoint=True, cline=False)

# Generate Command Trajectory
com_rot_queue = []
theta = np.linspace(0, 3.5*np.pi, 500)
t = theta * (1+theta**2)**0.5 + np.log((1+theta**2)**0.5 + theta)
f = interp1d(t, theta)
t_ = np.linspace(0, t[-1], 80)
theta_ = f(t_)
beta = 10 / 180 * np.pi / theta[-1]
for _ in range(10):
    com_rot_queue.append(np.eye(3))
for i in range(len(t_)):
    com_rot_queue.append(R.from_euler('zyz', [theta_[i], beta*theta_[i], -theta_[i]]).as_matrix())

grp_log, nob_log, obj_log, ct_log = [], [], [], []
tact_log, tact_gt_log, penetration_log, tact_est_log, tact_offset, tact_gt_offset = [], [], [], [], [], []
st_log = []
F_world_wr_online = []
F_gt = []

for i, comrot in enumerate(tqdm((touch_begin*[np.eye(3)] + com_rot_queue))):

    if i == touch_begin:
        graph.touch = True
        simulator.touch = True
    
    g_new = graph.G0_world.compose(graph.current_estimate.atPose3(G(graph.i+1)))
    cart_new_offset = np.hstack((g_new.translation(), R.from_matrix(g_new.rotation().matrix()).as_quat()))

    cart_new = tcp_coc(cart_new_offset, coc_offset)
    simulator.add_new(cart_new)
    
    tact_new = np.hstack((simulator.gg_.translation()+simulator.OU_noise[3:], R.from_matrix(simulator.gg_.rotation().matrix()).as_euler('zyx',True)+180/np.pi*simulator.OU_noise[:3]))
    tact_new_offset = tact_new.copy()
    tact_new_offset[:3] -= np.cross( R.from_euler('zyx', tact_new[3:], True).as_rotvec(), coc_offset )
    tact_new_gt = np.hstack((simulator.gg_.translation(), R.from_matrix(simulator.gg_.rotation().matrix()).as_euler('zyx',True)))
    tact_new_gt_offset = tact_new_gt.copy()
    tact_new_gt_offset[:3] -= np.cross( R.from_euler('zyx', tact_new_gt[3:], True).as_rotvec(), coc_offset )

    graph.add_new((cart_new_offset, tact_new_offset, comrot))

    rot = R.from_quat(cart_init[3:]).as_matrix() @ graph.current_estimate.atPose3(G(graph.i)).rotation().matrix()
    F_world_wr_online.append(rot @ graph.current_estimate.atVector(W(graph.i))[3:])
    F_gt.append(simulator.force)

    grp_log.append(np.hstack((simulator.grp.translation(), R.from_matrix(simulator.grp.rotation().matrix()).as_euler('zyx',True))))
    nob_log.append(np.hstack((simulator.nob.translation(), R.from_matrix(simulator.nob.rotation().matrix()).as_euler('zyx',True))))
    obj_log.append(np.hstack((simulator.obj.translation(), R.from_matrix(simulator.obj.rotation().matrix()).as_euler('zyx',True))))
    ct_log.append(np.hstack((simulator.ct.translation(), R.from_matrix(simulator.ct.rotation().matrix()).as_euler('zyx',True))))
    tact_log.append(np.hstack((simulator.gg_.translation()+simulator.OU_noise[3:], R.from_matrix(simulator.gg_.rotation().matrix()).as_euler('zyx',True)+180/np.pi*simulator.OU_noise[:3])))
    tact_gt_log.append(np.hstack((simulator.gg_.translation(), R.from_matrix(simulator.gg_.rotation().matrix()).as_euler('zyx',True))))
    tact_est_log.append(graph.tact_est)
    tact_offset.append(tact_new_offset)
    tact_gt_offset.append(tact_new_gt_offset)
    st_log.append(graph.st)

    if show_viz:
        viz.plot_clear()

        viz.plot_confidence_ellipsoid(graph.ct_.translation()-coc_offset, graph.ct_.rotation().matrix() @ graph.ct_cov[3:,3:] @ graph.ct.rotation().matrix().T)
        viz.plot_gripper(graph.grp_.translation()-coc_offset, graph.grp_.rotation().matrix())
        viz.plot_cpoint(graph.ct_.translation()-coc_offset, graph.ct_.rotation().matrix())
        viz.plot_object(graph.obj_.translation()-coc_offset, graph.obj_.rotation().matrix())

        viz.plot_coordinate_axis(simulator.obj.translation(), simulator.obj.rotation().matrix(), lw=4, c='r')
        viz.plot_coordinate_axis(simulator.nob.translation(), simulator.nob.rotation().matrix(), c='b')
        viz.plot_coordinate_axis(simulator.ct.translation(), simulator.ct.rotation().matrix(), c='k')
        viz.plot_update(simulator.grp.translation(), simulator.grp.rotation().matrix(),
                simulator.obj.translation(), simulator.obj.rotation().matrix(),
                simulator.ct.translation(), simulator.ct.rotation().matrix(),
                simulator.nob.translation(), simulator.nob.rotation().matrix(), alpha=.1)

graph.change_command_type('translation')
com_trn_queue = np.zeros((100,3))
com_trn_queue[:,1] = np.linspace(0,10,100)

traj_list = np.array(100*[cart_new_offset])
traj_list[:,1] += np.linspace(0,10,100)

control = 'open'

for i, comtrn in enumerate(tqdm(com_trn_queue)):

    if i == touch_begin:
        graph.touch = True
        simulator.touch = True
    
    if control == 'closed':
        g_new = graph.G0_world.compose(graph.current_estimate.atPose3(G(graph.i+1)))
        cart_new_offset = np.hstack((g_new.translation(), R.from_matrix(g_new.rotation().matrix()).as_quat()))
    elif control == 'open':
        cart_new_offset = traj_list[i]

    cart_new = tcp_coc(cart_new_offset, coc_offset)
    simulator.add_new(cart_new)
    
    tact_new = np.hstack((simulator.gg_.translation()+simulator.OU_noise[3:], R.from_matrix(simulator.gg_.rotation().matrix()).as_euler('zyx',True)+180/np.pi*simulator.OU_noise[:3]))
    tact_new_offset = tact_new.copy()
    tact_new_offset[:3] -= np.cross( R.from_euler('zyx', tact_new[3:], True).as_rotvec(), coc_offset )
    tact_new_gt = np.hstack((simulator.gg_.translation(), R.from_matrix(simulator.gg_.rotation().matrix()).as_euler('zyx',True)))
    tact_new_gt_offset = tact_new_gt.copy()
    tact_new_gt_offset[:3] -= np.cross( R.from_euler('zyx', tact_new_gt[3:], True).as_rotvec(), coc_offset )

    graph.add_new((cart_new_offset, tact_new_offset, comtrn))

    rot = R.from_quat(cart_init[3:]).as_matrix() @ graph.current_estimate.atPose3(G(graph.i)).rotation().matrix()
    F_world_wr_online.append(rot @ graph.current_estimate.atVector(W(graph.i))[3:])
    F_gt.append(simulator.force)

    grp_log.append(np.hstack((simulator.grp.translation(), R.from_matrix(simulator.grp.rotation().matrix()).as_euler('zyx',True))))
    nob_log.append(np.hstack((simulator.nob.translation(), R.from_matrix(simulator.nob.rotation().matrix()).as_euler('zyx',True))))
    obj_log.append(np.hstack((simulator.obj.translation(), R.from_matrix(simulator.obj.rotation().matrix()).as_euler('zyx',True))))
    ct_log.append(np.hstack((simulator.ct.translation(), R.from_matrix(simulator.ct.rotation().matrix()).as_euler('zyx',True))))
    tact_log.append(np.hstack((simulator.gg_.translation()+simulator.OU_noise[3:], R.from_matrix(simulator.gg_.rotation().matrix()).as_euler('zyx',True)+180/np.pi*simulator.OU_noise[:3])))
    tact_gt_log.append(np.hstack((simulator.gg_.translation(), R.from_matrix(simulator.gg_.rotation().matrix()).as_euler('zyx',True))))
    tact_est_log.append(graph.tact_est)
    tact_offset.append(tact_new_offset)
    tact_gt_offset.append(tact_new_gt_offset)
    st_log.append(graph.st)

    if show_viz:
        viz.plot_clear()

        viz.plot_confidence_ellipsoid(graph.ct_.translation()-coc_offset, graph.ct_.rotation().matrix() @ graph.ct_cov[3:,3:] @ graph.ct.rotation().matrix().T)
        viz.plot_gripper(graph.grp_.translation()-coc_offset, graph.grp_.rotation().matrix())
        viz.plot_cpoint(graph.ct_.translation()-coc_offset, graph.ct_.rotation().matrix())
        viz.plot_object(graph.obj_.translation()-coc_offset, graph.obj_.rotation().matrix())

        viz.plot_coordinate_axis(simulator.obj.translation(), simulator.obj.rotation().matrix(), lw=4, c='r')
        viz.plot_coordinate_axis(simulator.nob.translation(), simulator.nob.rotation().matrix(), c='b')
        viz.plot_coordinate_axis(simulator.ct.translation(), simulator.ct.rotation().matrix(), c='k')
        viz.plot_update(simulator.grp.translation(), simulator.grp.rotation().matrix(),
                simulator.obj.translation(), simulator.obj.rotation().matrix(),
                simulator.ct.translation(), simulator.ct.rotation().matrix(),
                simulator.nob.translation(), simulator.nob.rotation().matrix(), alpha=.1)

plt.figure()
plt.plot(tact_gt_offset)

plt.figure()
plt.plot(grp_log)

plt.figure()
plt.plot(np.array(F_gt))

plt.show()

input()