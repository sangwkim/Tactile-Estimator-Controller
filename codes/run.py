#!/usr/bin/env python

import os, rospy, threading, argparse, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime
from packing_environment import Robot_motion, Packing_env
from glob import glob
sys.path = sys.path[::-1] # This is just a trick to run Python2 (ROS) and Python3 (GTSAM) at the same time, if you are using newer ROS, you maybe don't have to do this.
import cv2

# Data saving function
def save_data(datas, cart_init, error_raise, times, grasp_pose, misalign,
                object_name, graspForce, rp, pivot_rot, edge, corner, gtsam_st, control, save_image=False):
    
    datas_ = []
    for data in datas:
        if data != None:
            datas_.append(data)

    if save_image:
        image_g1 = np.vstack([data[0] for data in datas_])
        image_g2 = np.vstack([data[1] for data in datas_])

    time_g1 = np.vstack([np.expand_dims(data[2],1) for data in datas_])
    time_g2 = np.vstack([np.expand_dims(data[3],1) for data in datas_])
    cart_g1 = np.vstack([data[4] for data in datas_])
    cart_g2 = np.vstack([data[5] for data in datas_])
    tact_g1 = np.vstack([data[6] for data in datas_])
    tact_g2 = np.vstack([data[7] for data in datas_])
    comrot_g1 = np.vstack([data[8] for data in datas_])
    ft = np.vstack([data[9] for data in datas_])
    if control == 'optimal':
        waypoints = np.vstack([data[10] for data in datas_])
        si = np.hstack([data[11] for data in datas_])
    added = np.hstack([data[12] for data in datas_])
    force_est = np.vstack([data[13] for data in datas_])
    wrench_est = np.vstack([data[14] for data in datas_])
    disp_est = np.vstack([data[15] for data in datas_])
    ct_est = np.vstack([data[16] for data in datas_])
    ct_cov = np.vstack([data[17] for data in datas_])

    save_folder = data_folder + '/' + object_name + '/' + datetime.now().strftime("%Y%m%d%H%M%S")
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)
    if not os.path.isdir(data_folder + '/' + object_name):
        os.mkdir(data_folder + '/' + object_name)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    if save_image:
        for i in range(len(image_g1)):
            cv2.imwrite(save_folder + '/g1_' + str(i) + '.jpg',
                        image_g1[i, :, :, :])
            cv2.imwrite(save_folder + '/g2_' + str(i) + '.jpg',
                        image_g2[i, :, :, :])

    np.save(save_folder + '/' + 'time_g1_rock.npy', time_g1)
    np.save(save_folder + '/' + 'time_g2_rock.npy', time_g2)
    np.save(save_folder + '/' + 'cart_g1_rock.npy', cart_g1)
    np.save(save_folder + '/' + 'cart_g2_rock.npy', cart_g2)
    np.save(save_folder + '/' + 'tact_g1_rock.npy', tact_g1)
    np.save(save_folder + '/' + 'tact_g2_rock.npy', tact_g2)
    np.save(save_folder + '/' + 'cart_init.npy', cart_init)
    np.save(save_folder + '/' + 'error_raise.npy', error_raise)
    np.save(save_folder + '/' + 'grasp_pose.npy', grasp_pose)
    np.save(save_folder + '/' + 'grasp_force.npy', graspForce)
    np.save(save_folder + '/' + 'misalign.npy', misalign)
    np.save(save_folder + '/' + 'timeflag.npy', times)
    np.save(save_folder + '/' + 'comrot_g1.npy', comrot_g1)
    np.save(save_folder + '/' + 'wrench.npy', ft)
    if control == 'optimal':
        np.save(save_folder + '/' + 'waypoints.npy', waypoints)
        np.save(save_folder + '/' + 'si.npy', si)
    np.save(save_folder + '/' + 'added.npy', added)
    np.save(save_folder + '/' + 'force_est.npy', force_est)
    np.save(save_folder + '/' + 'wrench_est.npy', wrench_est)
    np.save(save_folder + '/' + 'disp_est.npy', disp_est)
    np.save(save_folder + '/' + 'ct_est.npy', ct_est)
    np.save(save_folder + '/' + 'ct_cov.npy', ct_cov)
    np.save(save_folder + '/' + 'rp.npy', rp)
    np.save(save_folder + '/' + 'pivot_rot.npy', pivot_rot)
    np.save(save_folder + '/' + 'edge.npy', edge)
    np.save(save_folder + '/' + 'corner.npy', corner)
    np.save(save_folder + '/' + 'gtsam_st.npy', gtsam_st)

# Generates random parameters that determines the spiral motion trajectory, and following tilting direction if the final contact formation is line or patch
# See the supplementary video to see how the spiral motion looks like
# rp: determines the initial pose of the object [roll, pitch] relative to the world frame
# pivot_rot: determines the final angle of the spiral
# edge: provides ground truth on which side of the edge will be in contact if we transition from point to line contact
def generate_random_command(object_name, min_rev):
    
    min_rot = min_rev * 2*np.pi
    
    # Coordinates of the corners for each of the objects
    corners_dict = \
        {'rectangle': np.array([[-17.5,-25],[17.5,-25],[17.5,25],[-17.5,25]]),
         'hexagon': np.array([[15.156,-8.75],[15.156,8.75],[0,17.5],[-15.156,8.75],[-15.156,-8.75],[0,-17.5]]),
         'obj_1': np.array([[0,22.134],[35,7.630],[35,36.329],[12.167,41.509]]) - np.array([17.5,27]),
         'obj_3': np.array([[3.919,8.398],[30.839,23.806],[3.919,42.91]]) - np.array([17.7,26]),
         'obj_4': np.array([[0,15.615],[35,7.63],[35,43.870]]) - np.array([17.5,29]),
         'obj_5': np.array([[0,25,4.055],[35,0,0],[35,50,1.986]]) - np.array([17.5,25,0]),
         'obj_6': - np.array([[0,0],[35,14.079],[35,43.87],[0,32.352]]) + np.array([[17.5,26]]),
         'obj_7': np.array([[0,0],[35,0],[35,50],[0,50]]) - np.array([17.5,25]),
         'magna_1': np.array([[-17.5,-22],[17.5,-22],[17.5,22],[-17.5,22]]),
         'magna_2': np.array([[1*np.cos(th), 1*np.sin(th)] for th in np.arange(0,2*np.pi,0.1*np.pi)]),
         'magna_3': np.array([[12*np.cos(th), 12*np.sin(th)] for th in np.arange(0,2*np.pi,0.1*np.pi)]),
         'magna_4': np.array([[-17.68,-14.33],[0,-32],[17.68,-14.33],[17.68,14.33],[0,32],[-17.68,14.33]]),
         'magna_5': np.array([[1*np.cos(th), 1*np.sin(th)] for th in np.arange(0,2*np.pi,0.1*np.pi)]),
         'magna_6': np.array([[12*np.cos(th), 12*np.sin(th)] for th in np.arange(0,2*np.pi,0.1*np.pi)]),
         'magna_7': np.array([[-17.68,-14.33],[0,-32],[17.68,-14.33],[17.68,14.33],[0,32],[-17.68,14.33]]),
         'magna_8': np.array([[1*np.cos(th), 1*np.sin(th)] for th in np.arange(0,2*np.pi,0.1*np.pi)]),
         'magna_9': np.array([[-5.4,-14.4],[5.4,-14.4],[5.4,14.4],[-5.4,14.4]])}


    if object_name == 'circle' or object_name == 'ellipse':
        angle = np.random.uniform(0,2*np.pi)
        rp = np.sqrt(np.random.uniform(0.3**2, 0.4**2)) * np.array([np.cos(angle), np.sin(angle)])
    else:
        corners = corners_dict[object_name]
        if object_name == 'magna_4' or object_name == 'magna_7':
            corner_idx = np.random.choice([0,2,3,5])
            if corner_idx == 2 or corner_idx == 5:
                edge = 1
            else:
                edge = 0
        elif object_name == 'magna_9':
            corner_idx = np.random.randint(len(corners))
            if corner_idx == 0 or corner_idx == 2:
                edge = 0
            else:
                edge = 1
        else:
            corner_idx = np.random.randint(len(corners))
            edge = np.random.randint(2)
        e = 1 if edge else -1
        direction = corners[(corner_idx+e)%len(corners)] - corners[corner_idx]
        if 'magna' in object_name:
            pivot_rot = min_rot + np.arctan2(-direction[1], direction[0])%(2*np.pi)
        else:
            pivot_rot = min_rot + np.arctan2(-direction[1], direction[0])%(2*np.pi) + np.random.uniform(-0.11,0.11)*np.pi

        # If the bottom surface is slanted
        if len(corners[0]) == 3:
            normal = np.cross(corners[1] - corners[0], corners[2] - corners[1])
            normal /= np.linalg.norm(normal)
            rotvec = np.cross(np.array([0,0,1]), normal)
            rotvec *= np.arcsin(np.linalg.norm(rotvec)) / np.linalg.norm(rotvec)
            rp_0 = - np.array([rotvec[0], rotvec[1]])
        else:
            rp_0 = np.zeros(2)

        if final_cf == 'c':
            angle = np.random.uniform(-np.pi/24,np.pi/24)
        else:
            angle = np.random.uniform(-np.pi/6,np.pi/6)

        rp = np.array([0.3, 0]) + np.sqrt(np.random.uniform(0, 0.1**2)) * np.array([np.cos(angle), np.sin(angle)])
        a = corners[corner_idx] - corners[(corner_idx+1)%len(corners)]
        a /= np.linalg.norm(a)
        b = corners[corner_idx] - corners[(corner_idx-1)%len(corners)]
        b /= np.linalg.norm(b)
        c = a + b
        angle_add = np.arctan2(c[1],c[0]) + np.pi/2
        rp = rp_0 +  np.array([[np.cos(angle_add), -np.sin(angle_add)], [np.sin(angle_add), np.cos(angle_add)]]) @ rp

        if final_cf == 'c':
            pivot_rot = min_rot + np.arctan2(c[1],-c[0])%(2*np.pi)
            print(f"{pivot_rot}  {min_rot}  {np.arctan2(c[1],-c[0])%(2*np.pi)}")
            print(f"c: {c}, e: {e}, min_rot: {min_rot%(2*np.pi)/np.pi*180}, pivot_rot: {pivot_rot%(2*np.pi)/np.pi*180}, {np.arctan2(c[1],-c[0])/np.pi*180}")

    
    return rp, pivot_rot, edge, corners[corner_idx]

def main():

    object_dict = {
        'r': 'rectangle', 'h': 'hexagon', 'c': 'circle', 'e': 'ellipse',
        '1': 'obj_1', '3': 'obj_3', '4': 'obj_4', '5': 'obj_5', '6': 'obj_6', '7': 'obj_7',
        'v': 'magna_1', 'w': 'magna_2', 'x': 'magna_3', 'y': 'magna_4', 'z': 'magna_5', 't': 'magna_6', 'u': 'magna_7', 's': 'magna_8', 'q': 'magna_9'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, default='r', help='list of object shorthands')
    parser.add_argument('--dataset_name', type=str, default='noname')
    parser.add_argument('--env_type', type=str, default='floor', choice=['floor', 'bar'])
    parser.add_argument('--control', type=str, default='optimal', choice=['optimal', 'proportional'], help='type of controller to use. optimal: factor graph / proportional: constant deformation')
    parser.add_argument('--random_orientation', type=bool, default=False, help='random initial orientation. If false, just pre-specified orientation')
    parser.add_argument('--noenergyfactor', type=bool, default=False, help='If True, turns off the energy factor')
    parser.add_argument('--noonlinestiff', type=bool, default=False, help='If True, use fixed grasp parameters')
    parser.add_argument('--min_type', type=str, default='force', choices=['force'])
    parser.add_argument('--min_rev', type=int, default=4, help='minimum number of revolution for the spiral trajectory')
    parser.add_argument('--cone_angle', type=float, default=5, help='final angle of the spiral trajectory from the initial orientation')
    parser.add_argument('--cone_disc', type=int, default=100, help='number of discretization of the spiral trajectory')
    parser.add_argument('--final_cf', type=str, default='p', choices=['p','l','s','c']) # p: point / l: line / s: patch / c: point (large)
    parser.add_argument('--line_disc', type=int, default=100, help='number of discretization of the command trajectory after the line transition')
    parser.add_argument('--save_image', type=bool, default=False, help='If True, saves the tactile images')
    args = parser.parse_args()
    object_name = object_dict[vars(args)['object']]
    dataset_name = vars(args)['dataset_name']
    env_type = vars(args)['env_type']
    control = vars(args)['control']
    random_orientation = vars(args)['random_orientation']
    noenergyfactor = vars(args)['noenergyfactor']
    useonlinestiff = not vars(args)['noonlinestiff']
    min_type = vars(args)['min_type']
    min_rev = vars(args)['min_rev']
    cone_angle = vars(args)['cone_angle']
    cone_disc = vars(args)['cone_disc']
    global final_cf
    final_cf = vars(args)['final_cf']
    line_disc = vars(args)['line_disc']
    save_image = vars(args)['save_image']

    global data_folder
    data_folder = "/home/mcube/sangwoon/data/" + dataset_name

    # If repeat the previous dataset, load data from the original dataset.
    if 'repeat' in data_folder:
        data_folder_ = sorted(glob(f"{data_folder[:-7]}/{object_name}/2*/", recursive = False))[-1]
        gtsam_st_prior = np.load(data_folder_+'gtsam_st.npy')
    elif 'offlinesame' in data_folder:
        i = len(sorted(glob(f"{data_folder}/{object_name}/2*/", recursive = False)))
        data_folder_ = sorted(glob(f"{data_folder[:-12]}/{object_name}/2*/", recursive = False))[i]
        gtsam_st_prior = np.zeros(9)
    else:
        gtsam_st_prior = np.zeros(9)

    # Adjust prior based on approximate size of the object
    reach_dict = {'obj_4': 57, 'magna_1': -52, 'magna_2': 72, 'magna_4': 101, 'magna_5': 132, 'magna_6': 101, 'magna_7': 72, 'magna_8': 77, 'magna_9': 97}
    if object_name in list(reach_dict.keys()):
        reach = reach_dict[object_name]
    else:
        reach = 62

    env = Packing_env(env_type=env_type, gtsam_st_prior=gtsam_st_prior, reach=reach)
    env.target_object = object_name
    robot = Robot_motion(env_type=env_type)

    robot.open_gripper()
    robot.move_cart_add(0., 0., 50.)
    robot.setSpeed(600, 200)
    robot.robot_reset()
    
    if 'repeat' in data_folder or 'offlinesame' in data_folder:
        rp = np.load(data_folder_+'rp.npy')
        pivot_rot = np.load(data_folder_+'pivot_rot.npy')
        edge = np.load(data_folder_+'edge.npy')
        corner = np.load(data_folder_+'corner.npy')
    elif random_orientation:
        rp, pivot_rot, edge, corner = generate_random_command(object_name, min_rev)
    else: # if not random, pre-specified orientation
        if object_name == "rectangle":
            rp = np.array([0.3, 0.3]) # roll, pitch
            pivot_rot = min_rev*2*np.pi + 6/4*np.pi
            edge = 1
            corner = np.array([[17.5,-25]])
        elif object_name == 'hexagon':
            rp = np.array([0.5, 0])
            angle_add = 0 * np.pi/3
            rp = np.array([[np.cos(angle_add), -np.sin(angle_add)], [np.sin(angle_add), np.cos(angle_add)]]) @ rp
            pivot_rot = min_rev*2*np.pi + 7/6*np.pi
            edge = 0
            corner = np.array([0,-17.5])
        elif object_name == 'obj_1':
            rp = np.array([0.1,-0.4])
            pivot_rot = min_rev*2*np.pi + 0.125*np.pi
            edge = 1
            corner = np.array([-17.5,-4.866])
        else: 
            raise NotImplementedError

    if 'repeat' in data_folder or 'offlinesame' in data_folder:
        grasp_pose = np.load(data_folder_+'grasp_pose.npy')
    else:
        grasp_pose = np.zeros(6)
        if random_orientation:
            grasp_pose[:3] = np.random.uniform((0, 0, 0), (0, 0, 7.5))
        else:
            grasp_pose[:3] = 0
        grasp_pose[5] = -rp[0]
        # Adjust grasp pose on approximate size of the object
        grasp_pose[2] += reach-62
        
    # relative pose from object (bottom surface center) to gripper (center)
    T_og = np.eye(4)
    T_og[:3,:3] = R.from_euler('zyx', grasp_pose[3:]).as_matrix()
    T_og[:3,-1] = np.array([0,0,52]) + grasp_pose[:3]

    graspForce = np.random.uniform(10, 15)
    robot.height = 52+grasp_pose[2]
    print(f"grasp height: {52+grasp_pose[2]:.1f}, grasp force: {graspForce:.1f}")

    robot.pick_up_object(env.target_object, graspForce, False, grasp_pose, np.array([0,rp[1],-grasp_pose[-1]]), corner)
    
    state_full = np.random.uniform((0,0,0),(0,0,0)) # This variable is not currently used. Please ignore.

    datas, cart_init, times, error_raise, gtsam_st = env.step(state_full, grasp_pose, T_og, height=52+grasp_pose[2], pivot_rot=pivot_rot, edge=edge, line_disc=line_disc, cone_angle=cone_angle, cone_disc=cone_disc, final_cf=final_cf, ori=np.array([0,rp[1],-grasp_pose[-1]]), control=control, noenergyfactor=noenergyfactor, useonlinestiff=useonlinestiff, min_type=min_type)

    #if not env.tactile_module.error_raise and not env.tactile_module.safety_flag:
    try:
        thread = threading.Thread(target=save_data,
                                    args=(datas, cart_init, error_raise,
                                        times, grasp_pose.copy(), state_full.copy(),
                                        object_name, graspForce, rp, pivot_rot, edge, corner, gtsam_st, control, save_image))
        thread.start()
    except:
        pass
    
    object_cart_info = list(robot.objectCartesianDict[object_name])
    robot.return_object(object_cart_info[0], object_cart_info[1], grasp_pose)

if __name__ == '__main__':
    rospy.init_node('FG_host', anonymous=True)
    main()