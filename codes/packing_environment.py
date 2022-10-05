#!/usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from utils_viz import visualization
import numpy as np
import time
from visualization_msgs.msg import *
from wsg_50_common.srv import *
from robot_comm.srv import *
import sys
sys.path = sys.path[::-1]
import rospy, os, cv2
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import *
from std_msgs.msg import Float64MultiArray, Float64
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tactile_module import tactile_module
from collections import deque
from scipy.interpolate import interp1d
import gtsam


class Robot_motion:
    def __init__(self, env_type='floor', gt_collect=False):
        self.r_convert = R.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        self.TCP_offset = np.array([-6,0,-12.]) # This is the mismatch between the actual TCP and the measured cartesian
        self.env_type = env_type

        if not gt_collect:            
            self.initialCartesian = self.TCP_convert_inv([230, -310, 180, 0.7071, 0.7071, 0, 0])
            self.cartesianOfCircle = self.TCP_convert_inv([-14.5, -384.5, 48.97, 0.7071, 0.7071, 0, 0])
            self.cartesianOfEllipse = self.TCP_convert_inv([-14.7, -469.5, 48.97, 0.7071, 0.7071, 0, 0])
            self.cartesianOfRectangle = self.TCP_convert_inv([60, -350, 48.97, 0.7071, 0.7071, 0, 0])
            self.cartesianOfHexagon = self.TCP_convert_inv([60, -270, 49, 0.7071, 0.7071, 0, 0])
            self.cartesianOfObj1 = self.TCP_convert_inv([400, -190, 49, 0.7071, 0.7071, 0, 0])
            self.cartesianOfObj3 = self.TCP_convert_inv([60, -190, 49, 0.7071, 0.7071, 0, 0])
            self.cartesianOfObj4 = self.TCP_convert_inv([400, -270, 49, 0.7071, 0.7071, 0, 0])
            self.cartesianOfObj5 = self.TCP_convert_inv([400, -350, 54.5, 0.7071, 0.7071, 0, 0])
            self.cartesianOfObj6 = self.TCP_convert_inv([400, -430, 49, 0.7071, 0.7071, 0, 0])
            self.cartesianOfObj7 = self.TCP_convert_inv([60, -430, 49, 0.7071, 0.7071, 0, 0])
            self.cartesianOfMagna1 = self.TCP_convert_inv([-20, -190, 49+9, 0.7071, 0.7071, 0, 0])
            self.cartesianOfMagna2 = self.TCP_convert_inv([-20, -270, 49+3, 0.7071, 0.7071, 0, 0])
            self.cartesianOfMagna3 = self.TCP_convert_inv([-20, -350, 49, 0.7071, 0.7071, 0, 0])
            self.cartesianOfMagna4 = self.TCP_convert_inv([-20, -430, 49, 0.7071, 0.7071, 0, 0])
            self.cartesianOfMagna5 = self.TCP_convert_inv([-20, -510, 49+5, 0.7071, 0.7071, 0, 0])
            self.cartesianOfMagna6 = self.TCP_convert_inv([-20, -350, 49, 0.7071, 0.7071, 0, 0])
            self.cartesianOfMagna7 = self.TCP_convert_inv([-20, -430, 49, 0.7071, 0.7071, 0, 0])
            self.cartesianOfMagna8 = self.TCP_convert_inv([60, -510, 49+2, 0.7071, 0.7071, 0, 0])
            self.cartesianOfMagna9 = self.TCP_convert_inv([140, -510, 49, 0.7071, 0.7071, 0, 0])
        else:           
            self.initialCartesian = self.TCP_convert_inv([-66.76, -300, 200, 1, 0, 0, 0])
            self.cartesianOfCircle = self.TCP_convert_inv([-15, -255, 74, 1, 0, 0, 0])
            self.cartesianOfEllipse = self.TCP_convert_inv([-15, -330, 74, 1, 0, 0, 0])
            self.cartesianOfRectangle = self.TCP_convert_inv([-15, -255, 74, 1, 0, 0, 0])
            self.cartesianOfHexagon = self.TCP_convert_inv([-15, -330, 74, 1, 0, 0, 0])

        self.cartesianOfFloor = self.TCP_convert_inv([235, -320, 110 ,0.7071,0.7071,0,0])
        self.cartesianOfBar = self.TCP_convert_inv([235, -320, 190 ,0.7071,0.7071,0,0])

        self.cartesianOfCircle_top = list(self.cartesianOfCircle)
        self.cartesianOfCircle_top[2] += 100
        self.cartesianOfEllipse_top = list(self.cartesianOfEllipse)
        self.cartesianOfEllipse_top[2] += 100
        self.cartesianOfRectangle_top = list(self.cartesianOfRectangle)
        self.cartesianOfRectangle_top[2] += 100
        self.cartesianOfHexagon_top = list(self.cartesianOfHexagon)
        self.cartesianOfHexagon_top[2] += 100
        self.cartesianOfObj1_top = list(self.cartesianOfObj1)
        self.cartesianOfObj1_top[2] += 100
        self.cartesianOfObj3_top = list(self.cartesianOfObj3)
        self.cartesianOfObj3_top[2] += 100
        self.cartesianOfObj4_top = list(self.cartesianOfObj4)
        self.cartesianOfObj4_top[2] += 100
        self.cartesianOfObj5_top = list(self.cartesianOfObj5)
        self.cartesianOfObj5_top[2] += 100
        self.cartesianOfObj6_top = list(self.cartesianOfObj6)
        self.cartesianOfObj6_top[2] += 100
        self.cartesianOfObj7_top = list(self.cartesianOfObj7)
        self.cartesianOfObj7_top[2] += 100
        self.cartesianOfObj7_top[1] -= 10
        self.cartesianOfMagna1_top = list(self.cartesianOfMagna1)
        self.cartesianOfMagna1_top[2] += 100
        self.cartesianOfMagna2_top = list(self.cartesianOfMagna2)
        self.cartesianOfMagna2_top[2] += 100
        self.cartesianOfMagna3_top = list(self.cartesianOfMagna3)
        self.cartesianOfMagna3_top[2] += 100
        self.cartesianOfMagna4_top = list(self.cartesianOfMagna4)
        self.cartesianOfMagna4_top[2] += 100
        self.cartesianOfMagna5_top = list(self.cartesianOfMagna5)
        self.cartesianOfMagna5_top[2] += 150
        self.cartesianOfMagna6_top = list(self.cartesianOfMagna6)
        self.cartesianOfMagna6_top[2] += 100
        self.cartesianOfMagna7_top = list(self.cartesianOfMagna7)
        self.cartesianOfMagna7_top[2] += 100
        self.cartesianOfMagna6_top = list(self.cartesianOfMagna6)
        self.cartesianOfMagna6_top[2] += 100
        self.cartesianOfMagna7_top = list(self.cartesianOfMagna7)
        self.cartesianOfMagna7_top[2] += 100
        self.cartesianOfMagna8_top = list(self.cartesianOfMagna8)
        self.cartesianOfMagna8_top[2] += 100
        self.cartesianOfMagna9_top = list(self.cartesianOfMagna9)
        self.cartesianOfMagna9_top[2] += 100

        self.objectCartesianDict = {'circle':[self.cartesianOfCircle,self.cartesianOfCircle_top],\
                                   'rectangle':[self.cartesianOfRectangle,self.cartesianOfRectangle_top],\
                                   'hexagon':[self.cartesianOfHexagon,self.cartesianOfHexagon_top],\
                                   'ellipse':[self.cartesianOfEllipse,self.cartesianOfEllipse_top],\
                                   'obj_1':[self.cartesianOfObj1,self.cartesianOfObj1_top],\
                                   'obj_3':[self.cartesianOfObj3,self.cartesianOfObj3_top],\
                                   'obj_4':[self.cartesianOfObj4,self.cartesianOfObj4_top],\
                                   'obj_5':[self.cartesianOfObj5,self.cartesianOfObj5_top],\
                                   'obj_6':[self.cartesianOfObj6,self.cartesianOfObj6_top],\
                                   'obj_7':[self.cartesianOfObj7,self.cartesianOfObj7_top],\
                                   'magna_1':[self.cartesianOfMagna1,self.cartesianOfMagna1_top],\
                                   'magna_2':[self.cartesianOfMagna2,self.cartesianOfMagna2_top],\
                                   'magna_3':[self.cartesianOfMagna3,self.cartesianOfMagna3_top],\
                                   'magna_4':[self.cartesianOfMagna4,self.cartesianOfMagna4_top],\
                                   'magna_5':[self.cartesianOfMagna5,self.cartesianOfMagna5_top],\
                                   'magna_6':[self.cartesianOfMagna6,self.cartesianOfMagna6_top],\
                                   'magna_7':[self.cartesianOfMagna7,self.cartesianOfMagna7_top],\
                                   'magna_8':[self.cartesianOfMagna8,self.cartesianOfMagna8_top],\
                                   'magna_9':[self.cartesianOfMagna9,self.cartesianOfMagna9_top]}

        ###############################holes#######################################
        self.cartesianOfGap_hole_Rectangle = self.TCP_convert_inv([100, -316.5, 75.0, 0.70803, 0.70618, -0.00185, -0.00185])
        self.cartesianOfGap_hole_Circle = self.TCP_convert_inv([158.2, -341.7, 75.0, 0.71074, 0.70338, -0.00737, -0.00737])
        self.cartesianOfGap_hole_Hexagon = self.TCP_convert_inv([160, -289, 75.0, 0.71074, 0.70338, -0.00737, -0.00737])
        self.cartesianOfGap_hole_Ellipse = self.TCP_convert_inv([100.5, -317.3, 75.0, 0.71074, 0.70338, -0.00737, -0.00737])

        self.cartesianOfGap_hole_Dict = {
            'rectangle': self.cartesianOfGap_hole_Rectangle,
            'circle': self.cartesianOfGap_hole_Circle,
            'hexagon': self.cartesianOfGap_hole_Hexagon,
            'ellipse': self.cartesianOfGap_hole_Ellipse
        }

        self.object_width = {'circle': 40., 'rectangle': 40., 'hexagon': 34., 'ellipse': 40., 'obj_1': 35, 'obj_3': 35, 'obj_4': 35, 'obj_5': 35, 'obj_6': 35, 'obj_7': 35,\
                             'magna_1': 40., 'magna_2': 40., 'magna_3': 40., 'magna_4': 30., 'magna_5': 40., 'magna_6': 30., 'magna_7': 40., 'magna_8': 35., 'magna_9': 10.}

        self.Start_EGM = rospy.ServiceProxy('/robot2_ActivateEGM', robot_ActivateEGM)
        self.Stop_EGM = rospy.ServiceProxy('/robot2_StopEGM', robot_StopEGM)
        self.setSpeed = rospy.ServiceProxy('/robot2_SetSpeed', robot_SetSpeed)
        self.stiffness_pub_Rx = rospy.Publisher('/stiffness/Rx', Float64, queue_size=10)
        self.stiffness_pub_Ry = rospy.Publisher('/stiffness/Ry', Float64, queue_size=10)
        self.stiffness_pub_Rz = rospy.Publisher('/stiffness/Rz', Float64, queue_size=10)
        self.stiffness_pub_x = rospy.Publisher('/stiffness/x', Float64, queue_size=10)
        self.stiffness_pub_y = rospy.Publisher('/stiffness/y', Float64, queue_size=10)
        self.stiffness_pub_z = rospy.Publisher('/stiffness/z', Float64, queue_size=10)
        self.deform_raw_Rx = rospy.Publisher('/deform_raw/Rx', Float64, queue_size=10)
        self.deform_raw_Ry = rospy.Publisher('/deform_raw/Ry', Float64, queue_size=10)
        self.deform_raw_Rz = rospy.Publisher('/deform_raw/Rz', Float64, queue_size=10)
        self.deform_raw_x = rospy.Publisher('/deform_raw/x', Float64, queue_size=10)
        self.deform_raw_y = rospy.Publisher('/deform_raw/y', Float64, queue_size=10)
        self.deform_raw_z = rospy.Publisher('/deform_raw/z', Float64, queue_size=10)
        self.deform_est_Rx = rospy.Publisher('/deform_est/Rx', Float64, queue_size=10)
        self.deform_est_Ry = rospy.Publisher('/deform_est/Ry', Float64, queue_size=10)
        self.deform_est_Rz = rospy.Publisher('/deform_est/Rz', Float64, queue_size=10)
        self.deform_est_x = rospy.Publisher('/deform_est/x', Float64, queue_size=10)
        self.deform_est_y = rospy.Publisher('/deform_est/y', Float64, queue_size=10)
        self.deform_est_z = rospy.Publisher('/deform_est/z', Float64, queue_size=10)
        self.penetration_pub = rospy.Publisher('/penetration', Float64, queue_size=10)
        self.command_pose_pub = rospy.Publisher('/robot2_EGM/SetCartesian', PoseStamped, queue_size=10, latch=True)
        self.EGM_pose_buffer = deque(maxlen=500)
        self.clearBuffer = rospy.ServiceProxy('/robot2_ClearBuffer', robot_ClearBuffer)
        self.addBuffer = rospy.ServiceProxy('/robot2_AddBuffer', robot_AddBuffer)
        self.executeBuffer = rospy.ServiceProxy('/robot2_ExecuteBuffer', robot_ExecuteBuffer)
        self.setZone = rospy.ServiceProxy('/robot2_SetZone', robot_SetZone)
        self.cf_detect_1_pub = rospy.Publisher('/cf_detection_1', Float64, queue_size=1)
        self.cf_detect_2_pub = rospy.Publisher('/cf_detection_2', Float64, queue_size=1)

    def TCP_convert(self, position):
        pos_conv = np.array(position)
        r = (R.from_quat(position[3:])).as_matrix()
        pos_conv[:3] += r.dot(self.TCP_offset)
        pos_conv[3:] = (R.from_quat(pos_conv[3:]) * self.r_convert.inv()).as_quat()
        return pos_conv

    def TCP_convert_inv(self, position):
        position[3:] = (R.from_quat(position[3:]) * self.r_convert).as_quat()
        pos_conv = np.array(position)
        r = (R.from_quat(position[3:])).as_matrix()
        pos_conv[:3] += r.dot(-self.TCP_offset)
        return pos_conv

    def move_cart_mm(self, position): #TODO: do safety check before running
        position = self.TCP_convert(position)
        setCartRos = rospy.ServiceProxy('/robot2_SetCartesian', robot_SetCartesian)
        setCartRos(position[0], position[1], position[2], position[6],
                   position[3], position[4], position[5])

    def move_cart_series(self, cart_list, speed=[600, 200]):
        self.setZone(4)  # 0~4
        self.clearBuffer()
        self.setSpeed(speed[0], speed[1])
        for cart in cart_list:
            cart = self.TCP_convert(cart)
            self.addBuffer(cart[0], cart[1], cart[2], cart[6], cart[3],
                           cart[4], cart[5])
        self.executeBuffer()
        self.setZone(1)

    def move_cart_add(self, dx=0, dy=0, dz=0):
        #Define ros services
        getCartRos = rospy.ServiceProxy('/robot2_GetCartesian',
                                        robot_GetCartesian)
        setCartRos = rospy.ServiceProxy('/robot2_SetCartesian',
                                        robot_SetCartesian)
        #read current robot pose
        c = getCartRos()
        #move robot to new pose
        setCartRos(c.x + dx, c.y + dy, c.z + dz, c.q0, c.qx, c.qy, c.qz)

    def get_cart(self):
        getCartRos = rospy.ServiceProxy('/robot2_GetCartesian',
                                        robot_GetCartesian)
        c = getCartRos()
        c_array = self.TCP_convert_inv([c.x, c.y, c.z, c.qx, c.qy, c.qz, c.q0])
        return list(c_array)

    def get_jointangle(self):
        getJoint = rospy.ServiceProxy('/robot2_GetJoints', robot_GetJoints)
        angle = getJoint()
        return [angle.j1, angle.j2, angle.j3, angle.j4, angle.j5, angle.j6]

    def set_jointangle(self, angle):
        setJoint = rospy.ServiceProxy('/robot2_SetJoints', robot_SetJoints)
        setJoint(angle[0], angle[1], angle[2], angle[3], angle[4], angle[5])

    def robot_reset(self):
        print('go to the initial position')
        self.move_cart_mm(self.initialCartesian)

    def close_gripper_f(self, grasp_speed=50, grasp_force=10, width=40.):
        grasp = rospy.ServiceProxy('/wsg_50_driver/grasp', Move)
        self.ack()
        self.set_grip_force(grasp_force)
        time.sleep(0.1)
        error = grasp(width=width, speed=grasp_speed)
        time.sleep(0.5)

    def home_gripper(self):
        self.ack()
        home = rospy.ServiceProxy('/wsg_50_driver/homing', Empty)
        try:
            error = home()
        except:
            pass
        time.sleep(0.5)
        # print('error', error)

    def open_gripper(self):
        self.ack()
        release = rospy.ServiceProxy('/wsg_50_driver/move', Move)
        release(55.0, 10)
        time.sleep(0.5)

    def set_grip_force(self, val=5):
        set_force = rospy.ServiceProxy('/wsg_50_driver/set_force', Conf)
        error = set_force(val)
        time.sleep(0.2)

    def ack(self):
        srv = rospy.ServiceProxy('/wsg_50_driver/ack', Empty)
        error = srv()
        time.sleep(0.5)

    def tran_rotate_robot(self, targetCartesian, x, y, theta): #TODO: safety check before running
        targetCartesian[:3] += R.from_quat(targetCartesian[3:]).as_matrix().dot(np.array([x,y,0]))
        targetCartesian[3:] = (R.from_quat(targetCartesian[3:]) * R.from_euler('z', theta, degrees=True)).as_quat()
        return targetCartesian

    def grasp_object(self, target_object, graspForce, inposition, random_pose, lift=False):
        object_cart_info = list(self.objectCartesianDict[target_object])
        objectCartesian = np.array(object_cart_info[0]).copy()
        objectCartesian[:3] += R.from_quat(objectCartesian[3:]).as_matrix().dot(random_pose[:3])
        objectCartesian[3:] = (R.from_quat(objectCartesian[3:]) * R.from_euler('zyx',random_pose[3:])).as_quat()

        objectCartesian_top = np.array(object_cart_info[1]).copy()
        objectCartesian_top[:3] += R.from_quat(objectCartesian_top[3:]).as_matrix().dot(random_pose[:3])
        objectCartesian_top[3:] = (R.from_quat(objectCartesian_top[3:]) * R.from_euler('zyx',random_pose[3:])).as_quat()

        self.setSpeed(600, 200)
        if not inposition:
            self.move_cart_mm(objectCartesian_top)
            time.sleep(0.5)
        self.move_cart_mm(objectCartesian)
        time.sleep(0.5)
        #input('press enter to continue')
        self.close_gripper_f(100, graspForce, self.object_width[target_object])
        time.sleep(0.5)
        if lift:
            self.move_cart_mm(objectCartesian_top)
            time.sleep(0.5)            

    # Pick up the target object and place on top of the environment
    # Height position is calculated to have a little gap between the environment and the object at the beginning
    def pick_up_object(self, target_object, graspForce, inposition, random_pose, ori=None, corner=None):
        self.grasp_object(target_object, graspForce, inposition, random_pose, lift=True)
        if self.env_type == 'floor':
            cartesianOfGap = self.cartesianOfFloor.copy()
        elif self.env_type == 'bar':
            cartesianOfGap = self.cartesianOfBar.copy()
        else:
            raise NotImplementedError('unknown environment type')
        if target_object == 'circle' and random_pose[1] == 0:
            rv = R.from_euler('zyx',ori).as_rotvec()
            rv /= np.linalg.norm(rv)
            vert = np.abs(R.from_euler('zyx',ori).as_matrix().dot(np.array([17.5*rv[1],-17.5*rv[0],-self.height]))[2] + self.height) + 0
            cartesianOfGap[:3] += R.from_quat(cartesianOfGap[3:]).as_matrix().dot(np.array([0,0,vert]))
            cartesianOfGap[3:] = (R.from_quat(cartesianOfGap[3:]) * R.from_euler('zyx',ori)).as_quat()
        elif target_object == 'ellipse' and random_pose[1] == 0:
            a = 25.
            b = 17.5
            rv = R.from_euler('zyx',ori).as_rotvec()
            rv /= np.linalg.norm(rv)
            vert = np.abs(R.from_euler('zyx',ori).as_matrix().dot(np.array([rv[1]*a**2/(a**2*rv[1]**2+b**2*rv[0]**2)**0.5,-rv[0]*b**2/(a**2*rv[1]**2+b**2*rv[0]**2)**0.5,-self.height]))[2] + self.height)
            cartesianOfGap[:3] += R.from_quat(cartesianOfGap[3:]).as_matrix().dot(np.array([0,0,vert]))
            cartesianOfGap[3:] = (R.from_quat(cartesianOfGap[3:]) * R.from_euler('zyx',ori)).as_quat()
        elif random_pose[1] == 0:
            if 'magna' in target_object:
                vert = np.abs(R.from_euler('zyx',ori).as_matrix().dot(np.array([corner[0],corner[1],-self.height]))[2] + self.height) + 3
            elif len(corner) == 2:
                vert = np.abs(R.from_euler('zyx',ori).as_matrix().dot(np.array([corner[0],corner[1],-self.height]))[2] + self.height) + 1.5
            else:
                vert = - R.from_euler('zyx',ori).as_matrix().dot(np.array([corner[0],corner[1],-self.height+corner[2]]))[2] - self.height + 1.5
            cartesianOfGap[:3] += R.from_quat(cartesianOfGap[3:]).as_matrix().dot(np.array([0,0,vert]))
            cartesianOfGap[3:] = (R.from_quat(cartesianOfGap[3:]) * R.from_euler('zyx',ori)).as_quat()
        cartesianOfGap[:3] += R.from_quat(cartesianOfGap[3:]).as_matrix().dot(random_pose[:3])
        cartesianOfGap[3:] = (R.from_quat(cartesianOfGap[3:]) * R.from_euler('zyx',random_pose[3:])).as_quat()
        cart_positon_top = cartesianOfGap.copy()
        cart_positon_top[:3] += R.from_quat(cart_positon_top[3:]).as_matrix().dot(np.array([0,0,25]))
        self.move_cart_mm(cart_positon_top)
        time.sleep(1.)
        self.move_cart_mm(cartesianOfGap)
        time.sleep(.5)

    def return_object(self, objectCartesian, objectCartesian_top, random_pose):

        objectCartesian = np.array(objectCartesian)
        objectCartesian_top = np.array(objectCartesian_top)

        objectCartesian[:3] += R.from_quat(objectCartesian[3:]).as_matrix().dot(random_pose[:3])
        objectCartesian[3:] = (R.from_quat(objectCartesian[3:]) * R.from_euler('zyx',random_pose[3:])).as_quat()

        objectCartesian_top[:3] += R.from_quat(objectCartesian_top[3:]).as_matrix().dot(random_pose[:3])
        objectCartesian_top[3:] = (R.from_quat(objectCartesian_top[3:]) * R.from_euler('zyx',random_pose[3:])).as_quat()

        self.setSpeed(600, 200)
        self.move_cart_add(0., 0., 100.)
        time.sleep(0.5)

        self.move_cart_mm(objectCartesian_top)
        time.sleep(0.5)

        self.setSpeed(100, 50)

        self.move_cart_mm(objectCartesian)
        time.sleep(0.5)

        self.open_gripper()
        time.sleep(0.2)
        
        self.setSpeed(600, 200)
        self.move_cart_add(0., 0., 100.)
        time.sleep(0.5)

    def EGM_command(self, cart, tactile_module=None):
        xyzquat = self.TCP_convert(cart)
        pose = PoseStamped()
        pose.pose.position.x = xyzquat[0]
        pose.pose.position.y = xyzquat[1]
        pose.pose.position.z = xyzquat[2]
        pose.pose.orientation.x = xyzquat[3]
        pose.pose.orientation.y = xyzquat[4]
        pose.pose.orientation.z = xyzquat[5]
        pose.pose.orientation.w = xyzquat[6]
        self.command_pose_pub.publish(pose)

    # proportional control to keep the deformation constant
    def proportional_control(self,
            tactile_module,
            Kpz= 15e-3,#.2e-1,
            Kpxs = 5e-2,#.5e-1,
            Kpys = 5e-2,#.5e-1,
            EGM_rate = 100,
            max_angle = 5 / 180 * np.pi, #4 / 180 * np.pi, 9
            pivot_rot =9.65*np.pi, #7.5*np.pi, 18
            t_num = 100 * 100 # number of timesteps: 100 timesteps / 1 sec
        ):
        
        tactile_module.openloop_mode = True
        tactile_module.new_thres = 0.5
        time.sleep(1)

        self.cart = self.get_cart()

        # computes timed trajectory
        theta = np.linspace(0, pivot_rot, 500)
        t = theta * (1+theta**2)**0.5 + np.log((1+theta**2)**0.5 + theta)
        f = interp1d(t, theta)
        t_ = np.linspace(0, t[-1], t_num)
        theta_ = f(t_)
        beta = max_angle / theta[-1]
        rot_traj = []
        for i in range(t_num):
            rot_traj.append(R.from_quat(self.cart[3:]) * R.from_euler('zyz', [theta_[i], beta*theta_[i], -theta_[i]]))
        c = beta * theta_[-1]
        a = 2 * c
        b = (a**2-c**2)**0.5
        t_num = 10000
        for i in range(t_num):
            yaw = theta_[-1] + 2*np.pi/t_num*i
            pitch = b**2 / (a + c*np.cos(2*np.pi/t_num*i))
            rot_traj.append(R.from_quat(self.cart[3:]) * R.from_euler('zyz', [yaw, pitch, -yaw]))

        # starts EGM controller
        try: ret = self.Stop_EGM()
        except: pass
        time.sleep(.5)
        for _ in range(100): self.EGM_command(self.cart)
        time.sleep(.1)
        for _ in range(100): self.EGM_command(self.cart)
        print("entering EGM: Point Pivot")
        ret = self.Start_EGM(False, 86400)
        for _ in range(100): self.EGM_command(self.cart)
        rate = rospy.Rate(EGM_rate)
        time.sleep(1)

        [dx_ema, dy_ema, dz_ema, dyaw_ema, dpit_ema, drol_ema] = tactile_module.nn_output_ema
        dz_ema = -dz_ema
        dyaw_ema = -dyaw_ema
        dz_set = dz_ema.copy()
        drol_set = drol_ema.copy()
        dpit_set = dpit_ema.copy()

        time_pp_b = time.time()

        l = len(rot_traj)-1

        for i in range(l):

            if tactile_module.safety_flag:
                error_raise = True
                print("too much force! safety stop! restarting")
                break

            ii = i + 1
            self.cart[3:] = rot_traj[ii].as_quat()
            cart_ct = tactile_module.cart_init[:3] + R.from_quat(tactile_module.cart_init[3:]).as_matrix().dot(tactile_module.ct_.translation())
            d_rotvec = rot_traj[ii-1].as_matrix().dot((rot_traj[ii-1].inv() * rot_traj[ii]).as_rotvec())
            self.cart[:3] += R.from_rotvec(d_rotvec).as_matrix().dot(self.cart[:3] - cart_ct) - (self.cart[:3] - cart_ct)

            [dx, dy, dz, dyaw, dpit, drol] = tactile_module.nn_output
            [dx_ema, dy_ema, dz_ema, dyaw_ema, dpit_ema, drol_ema] = tactile_module.nn_output_ema
            dz = -dz
            dyaw = -dyaw
            dz_ema = -dz_ema
            dyaw_ema = -dyaw_ema

            # positional gain to keep the deformation constant
            x_dot = Kpxs * (dpit-dpit_set)
            y_dot = -Kpys * (drol-drol_set)
            z_dot = - Kpz * (dz_set - dz)
            print(f"{x_dot:.6f}, {y_dot:.6f}, {z_dot:.6f}")

            self.cart[:3] += R.from_quat(self.cart[3:]).as_matrix()[:,0] * x_dot
            self.cart[:3] += R.from_quat(self.cart[3:]).as_matrix()[:,1] * y_dot
            self.cart[:3] += R.from_quat(self.cart[3:]).as_matrix()[:,2] * z_dot

            self.EGM_command(self.cart)
            rate.sleep()
            
        time_pp_f = time.time()

        # stops EGM controller
        time.sleep(.5)
        ret = self.Stop_EGM()
        print("stopping EGM: Point Pivot")
        time.sleep(.5)

        tactile_module.openloop_mode = False

        return time_pp_b, time_pp_f

    # move down straight until it detects touch
    def openloop_down(
            self,
            tactile_module,
            speed = .4,
            EGM_rate = 100,
            ):

        tactile_module.openloop_mode = True
        tactile_module.new_thres = 0.04

        tactile_module.restart_gtsam = True
        time.sleep(0.4)

        time.sleep(.5)

        self.cart = self.get_cart()

        try: ret = self.Stop_EGM()
        except: pass
        time.sleep(.5)
        for _ in range(100): self.EGM_command(self.cart)
        print(f"Entering EGM: open loop straight down")
        ret = self.Start_EGM(False, 86400)
        for _ in range(100): self.EGM_command(self.cart)
        rate = rospy.Rate(EGM_rate)
        time.sleep(1)

        tactile_module.restart_gtsam = True
        time.sleep(0.4)

        t_b = time.time()
        touch = False
        i, j = 0, 0
        while (not touch or i-j<60) and i < 10000:
            #print(i-j)
            if not touch and np.linalg.norm(tactile_module.nn_output / np.array([0.05, 0.05, 0.05, .2, .2, .2])) > 1:
                touch = True
                j = i
            i += 1
            self.cart[2] -= speed/EGM_rate

            self.EGM_command(self.cart)
            tactile_module.cart_command = self.cart.copy()
            rate.sleep()

        t_f = time.time()

        time.sleep(.5)
        ret = self.Stop_EGM()
        print(f"Stopping EGM: open loop straight down")
        time.sleep(.5)
        
        return t_b, t_f

    # Runs primitives like 'pushdown' or 'pivot'
    def run_primitive(
            self,
            tactile_module, 
            primitive, # 'pushdown' or 'pivot'
            EGM_rate = 100,
            display_on = True,
            display_interval = 250,
            t_num = 100, #200, 500
            max_angle = 5 / 180 * np.pi,
            pivot_rot = 9.6*np.pi,
            target_object = 'rectangle',
            edge = None,
            line_disc = None,
            final_cf = 'p'
            ):

        if display_on and primitive == 'pushdown':
            self.display_init(target_object)
        time.sleep(1)

        self.cart = self.get_cart()
        tactile_module.cart_command = self.cart.copy()

        try: ret = self.Stop_EGM()
        except: pass
        time.sleep(.5)
        for _ in range(100): self.EGM_command(self.cart)
        time.sleep(.1)
        print(f"Entering EGM: {primitive}")
        ret = self.Start_EGM(False, 86400)
        for _ in range(100): self.EGM_command(self.cart)
        rate = rospy.Rate(EGM_rate)
        time.sleep(1)

        if primitive == 'pushdown':
            tactile_module.restart_gtsam = True
            time.sleep(.6)

        if primitive == 'pushdown':
            # for pushdown, just command no rotation, then the controller will naturally push it down to get the minimum penetration.
            for _ in range(4000):
                tactile_module.com_rot_queue.append(np.eye(3))
        elif primitive == 'pivot':
            # for point pivot, command rotation to follow the spiral cone trajectory, then move to one direction until the line touch is made.
            # the below is just a equation for drawing that trajectory.
            tactile_module.thres = 0.6
            theta = np.linspace(0, pivot_rot, 500)
            t = theta * (1+theta**2)**0.5 + np.log((1+theta**2)**0.5 + theta)
            f = interp1d(t, theta)
            t_ = np.linspace(0, t[-1], t_num)
            theta_ = f(t_)
            beta = max_angle / theta[-1]
            for i in range(t_num):
                tactile_module.com_rot_queue.append(R.from_euler('zyz', [theta_[i], beta*theta_[i], -theta_[i]]).as_matrix())
            if final_cf == 'l' or final_cf == 's': # if the final (goal) contact formation is not a point
                for i in range(250):
                    tactile_module.com_rot_queue.append(R.from_euler('zyz', [theta_[-1], beta*theta_[-1]+0.0025*i, -theta_[-1]]).as_matrix())
            elif final_cf == 'c': # experiment 1-2: large cone
                c = beta * theta_[-1]
                a = 2 * c
                b = (a**2-c**2)**0.5
                for i in range(200):
                    yaw = theta_[-1] + 2*np.pi/200*i
                    pitch = b**2 / (a + c*np.cos(2*np.pi/200*i))
                    tactile_module.com_rot_queue.append(R.from_euler('zyz', [yaw, pitch, -yaw]).as_matrix())

        t_b = time.time()
        i = 0
        gtsam_it = 0
        cf = 0 # point: 0 / line: 1 / patch: 2
        
        error_raise = False
        while len(tactile_module.com_rot_queue) != 0: # run the controller until the command rotation queue is empty
            if tactile_module.error_raise:
                error_raise = True
                print("indeterminant error raised! restarting")
                break
            elif tactile_module.safety_flag:
                error_raise = True
                print("too much force! safety stop! restarting")
                break

            # If transition is detected from point to line contact
            if cf == 0 and tactile_module.cf == 1:

                if final_cf == 'p' or final_cf == 'c': break

                # if transition from point to line is detected,
                # re-draw the command trajectory to follow trajectory that looks like ax*sin(x) graph
                # the below code is just a equation for it.
                cf = 1
                gtsam_it = tactile_module.gtsam_i
                
                V = (R.from_matrix(tactile_module.com_rot_queue[0]).inv() * R.from_matrix(tactile_module.com_rot_queue[1])).as_rotvec()[:2]
                V /= np.linalg.norm(V)
                R_UV = np.array([[V[1],V[0]],[-V[0],V[1]]])
                arclen = 200 / 180 * np.pi
                discretization = 500
                dl = arclen / discretization
                omega = 2*np.pi / (1 / 180 * np.pi)
                u, v = 0, 0
                a = 3
                com_rot_queue_new = []
                for i in range(line_disc):# range(discretization):
                    u += dl / (1+(a*np.sin(omega*u)+a*omega*u*np.cos(omega*u))**2)**0.5
                    v = a * u * np.sin(omega*u)
                    xy = R_UV @ np.array([u,v]).T
                    rotvec = np.array([-xy[1],xy[0],0])
                    com_rot_queue_new.append(R.from_rotvec(rotvec).as_matrix())
                if final_cf == 's':
                    for i in range(300):
                        if edge == 0:
                            v -= dl
                        else:
                            v += dl
                        xy = R_UV @ np.array([u,v]).T
                        rotvec = np.array([-xy[1],xy[0],0])
                        com_rot_queue_new.append(R.from_rotvec(rotvec).as_matrix())

                tactile_module.com_rot_queue = 10 * [np.eye(3)] + com_rot_queue_new

            elif cf == 1 and tactile_module.cf == 2:
                # if transition from line to patch is detected, just command zero rotation to stablize the patch contact
                cf = 2
                print("plane detected! finishing manipulation")
                break
                
            # adjust the speed
            if np.all(tactile_module.com_rot_queue[0] == np.eye(3)):
                speed = 0.3
            elif cf == 0 and final_cf == 'c':
                speed = 1.
            elif cf == 0 and tactile_module.gtsam_i < t_num+30:
                speed = 1.
            elif cf == 0 and tactile_module.gtsam_i >= t_num+30:
                speed = 0.3
            elif cf == 1 and tactile_module.gtsam_i > gtsam_it + 50:
                speed = 0.5
            elif cf == 1 and tactile_module.gtsam_i <= gtsam_it + 50:
                speed = 1.
            else:
                speed = 0.5

            # turn on the contact formation transition (point -> line) detector when it starts decending towards the line contact
            if cf == 0 and (final_cf == 'l' or final_cf == 's') and tactile_module.gtsam_i > t_num+40:
                self.cf_detect_1_pub.publish(1)
            elif cf == 0 and (final_cf == 'l' or final_cf == 's') and tactile_module.gtsam_i > t_num+20 and ('magna' in target_object):
                self.cf_detect_1_pub.publish(1)
            # turn on the contact formation transition (line -> patch) detector
            elif cf == 1 and final_cf == 's' and tactile_module.gtsam_i > gtsam_it + 80:
                self.cf_detect_2_pub.publish(2)

            if primitive == "pushdown" and tactile_module.gtsam_i > 10:
                # if the object is touched and some number of timestep has passed, then stop the pushdown by emptying the command rotation queue.
                tactile_module.com_rot_queue = []
            
            i += 1
            
            # display the current estimate
            if display_on and i%display_interval==0: 
                self.display_update(tactile_module)

            # tactile_module.s is the length from the most recent added gripper pose on the interpolated planned path.
            if not tactile_module.transition:
                tactile_module.s += speed / EGM_rate
            try: si = np.max(np.where(np.array(tactile_module.s_list) - tactile_module.s < 0)[0])
            except: si = 0
            # if the speed is too fast so it already reached to the end of the planned trajectory,
            # then just stop at the last pose of the control horizon
            if si > len(tactile_module.waypoints)-2:
                pose = tactile_module.waypoints[-1]
            # otherwise, just move along the interpolated planned trajectory until the update on the factor graph module is over and ready to receive new measurements.
            else:
                p = tactile_module.waypoints[si]
                p_ = tactile_module.waypoints[si+1]
                lm = gtsam.Pose3.Logmap(p.inverse().compose(p_))
                pose = p.compose(gtsam.Pose3.Expmap(lm*(tactile_module.s-tactile_module.s_list[si])/(tactile_module.s_list[si+1]-tactile_module.s_list[si])))

            self.cart[:3] = pose.translation()
            self.cart[3:] = R.from_matrix(pose.rotation().matrix()).as_quat()

            self.EGM_command(self.cart)
            tactile_module.cart_command = self.cart.copy()
            rate.sleep()

        t_f = time.time()
        time.sleep(.5)
        ret = self.Stop_EGM()
        print(f"Stopping EGM: {primitive}")
        time.sleep(.5)
        
        return t_b, t_f, error_raise

    def display_init(self, target_object):
        pose_rel = np.zeros(6)
        pose_rel[:3] = R.from_quat(self.cart_init[3:]).as_matrix().dot(np.array([self.pose_error[0], self.pose_error[1], 0]))
        pose_rel[3:] = R.from_quat(self.cart_init[3:]).as_euler('zyx')
        self.viz = visualization(54.5, pose_rel, self.height, self.T_og, view_elev=30, view_azim=215,\
                            view_center=(0,0,-self.height), view_radius=40, env_type=self.env_type, object_name=target_object)
        self.viz.set_show(env=False, grp=True, obj_gt=True, obj_est=True, cpoint=True, cline=False)

    def display_update(self, tactile_module):
        self.viz.plot_clear()

        T_wg0 = np.eye(4)
        T_wg0[:3,:3] = R.from_quat(self.cart_init[3:]).as_matrix()
        T_wg0[:3,-1] = self.cart_init[:3]
        T_wg = np.eye(4)
        T_wg[:3,:3] = R.from_quat(tactile_module.cart_EGM[3:]).as_matrix()
        T_wg[:3,-1] = tactile_module.cart_EGM[:3]
        T_gd = np.eye(4)
        T_gd[:3,:3] = R.from_euler('zyx', tactile_module.nn_output[3:], degrees=True).as_matrix()
        T_gd[:3,-1] = tactile_module.nn_output[:3]

        T_g0g = np.linalg.inv(T_wg0) @ T_wg
        T_g0o = np.linalg.inv(T_wg0) @ T_wg @ T_gd @ np.linalg.inv(self.T_og)

        grp_gt_rot = T_g0g[:3,:3]
        grp_gt_trn = T_g0g[:3,-1]
        obj_gt_rot = T_g0o[:3,:3]
        obj_gt_trn = T_g0o[:3,-1]

        self.viz.plot_confidence_ellipsoid(tactile_module.ct_.translation(),\
                                      tactile_module.ct_.rotation().matrix()\
                                        @ tactile_module.ct_cov[3:,3:]\
                                        @ tactile_module.ct_.rotation().matrix().T)
        if tactile_module.cf == 1:
            self.viz.plot_confidence_cone(tactile_module.ct_.translation(), tactile_module.ct_.rotation().matrix(), tactile_module.ct_cov[:3,:3])
            self.viz.set_show(env=False, grp=True, obj_gt=True, obj_est=True, cpoint=True, cline=True)
        self.viz.plot_update(grp_gt_trn, grp_gt_rot,
                   tactile_module.obj_.translation(), tactile_module.obj_.rotation().matrix(),
                   tactile_module.ct_.translation(), tactile_module.ct_.rotation().matrix(),
                   obj_gt_trn, obj_gt_rot, alpha=.1)

class Packing_env:

    def __init__(self, env_type='floor', gt_collect=False, gtsam_st_prior=np.zeros(9), reach=62):
        
        self.env_type = env_type
        self.robot = Robot_motion(env_type=env_type)
        self.tactile_module = tactile_module(self.robot.TCP_offset, self.robot.r_convert, gtsam_st_prior, reach, verbose=False, gt_collect=gt_collect)
        self.target_object = None

    # parse the collected data
    def parse_data(self):

        image_g1, image_g2, time_g1, time_g2, cart_g1, cart_g2, tact_g1, tact_g2, comrot_g1, ft, waypoints,\
            si, added, force_est, wrench_est, disp_est, ct_est, ct_cov  = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for i in range(len(self.data1)):
            j = np.argmin(np.abs(np.array([d[1] for d in self.data2]) - self.data1[i][1]))
            image_g1.append(self.data1[i][0])
            image_g2.append(self.data2[j][0])
            time_g1.append(self.data1[i][1])
            time_g2.append(self.data2[j][1])
            comrot_g1.append(self.data1[i][2])
            cart_g1.append(self.data1[i][3])
            cart_g2.append(self.data2[j][3])
            tact_g1.append(self.data1[i][4])
            tact_g2.append(self.data2[j][4])
            ft.append(self.data1[i][5])
            waypoints.append(self.data1[i][6])
            si.append(self.data1[i][7])
            added.append(self.data1[i][8])
            force_est.append(self.data1[i][9])
            wrench_est.append(self.data1[i][10])
            disp_est.append(self.data1[i][11])
            ct_est.append(self.data1[i][12])
            ct_cov.append(self.data1[i][13])

        image_g1 = np.array(image_g1)
        image_g2 = np.array(image_g2)
        time_g1 = np.array(time_g1)
        time_g2 = np.array(time_g2)
        comrot_g1 = np.array(comrot_g1)
        cart_g1 = np.array(cart_g1)
        cart_g2 = np.array(cart_g2)
        cart_g1 = cart_g1.astype(np.float)
        cart_g2 = cart_g2.astype(np.float)
        tact_g1 = np.array(tact_g1)
        tact_g2 = np.array(tact_g2)
        ft = np.array(ft)
        waypoints = np.array(waypoints)
        si = np.array(si)
        added = np.array(added)
        force_est, wrench_est, disp_est, ct_est, ct_cov = np.array(force_est), np.array(wrench_est), np.array(disp_est), np.array(ct_est), np.array(ct_cov)

        return image_g1, image_g2, time_g1, time_g2, cart_g1, cart_g2, tact_g1, tact_g2, comrot_g1, ft, waypoints, si, added, force_est, wrench_est, disp_est, ct_est, ct_cov, self.tactile_module.cart_init

    def step(self, pose_error, rand_pose, T_og, height, pivot_rot, edge, line_disc, cone_angle, cone_disc, final_cf, ori=None, control='optimal', noenergyfactor=False, useonlinestiff=True, min_type='force'):

        self.robot.pose_error = pose_error
        self.robot.height = height
        self.robot.T_og = T_og

        self.robot.setSpeed(80, 40)

        self.tactile_module.data1.clear()
        self.tactile_module.data2.clear()
        self.tactile_module.noenergyfactor = noenergyfactor
        self.tactile_module.useonlinestiff = useonlinestiff
        self.tactile_module.min_type = min_type
        self.tactile_module.restart1 = True
        self.tactile_module.restart2 = True
        time.sleep(2)
        self.tactile_module.offset_calibrate = True
        time.sleep(.5)
        
        cart_init = self.robot.get_cart()
        self.robot.cart_init = cart_init.copy()

        display_on = False

        if control == 'optimal':
            time_pd_b, time_pd_f, _ = self.robot.run_primitive(self.tactile_module, 'pushdown', display_on=display_on, target_object=self.target_object)
        if control == 'proportional':
            time_pd_b, time_pd_f = self.robot.openloop_down(self.tactile_module)
        self.data1 = list(self.tactile_module.data1)[:]
        self.data2 = list(self.tactile_module.data2)[:]
        data_pushdown = self.parse_data()

        self.tactile_module.data1.clear()
        self.tactile_module.data2.clear()
        if control == 'optimal':
            time_pv_b, time_pv_f, error_raise = self.robot.run_primitive(self.tactile_module, 'pivot', speed=1.5, display_on=display_on, max_angle=cone_angle/180*np.pi, t_num=cone_disc, pivot_rot=pivot_rot, target_object=self.target_object, edge=edge, line_disc=line_disc, final_cf=final_cf)
        elif control == 'proportional':
            time_pv_b, time_pv_f = self.robot.proportional_control(self.tactile_module, display_on=display_on, max_angle=cone_angle/180*np.pi, pivot_rot=pivot_rot)
            error_raise, early_late = False, False

        self.data1 = list(self.tactile_module.data1)[:]
        self.data2 = list(self.tactile_module.data2)[:]
        data_pivot = self.parse_data()

        self.robot.move_cart_add(0., 0., 6.)
        time.sleep(0.5)

        return [data_pushdown, data_pivot], cart_init, [time_pd_b, time_pd_f, time_pv_b, time_pv_f], error_raise, self.tactile_module.st