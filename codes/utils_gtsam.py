#!/usr/bin/env python

import numpy as np
import gtsam
from gtsam_custom_factors import DispDiff, PoseDIff, TorqPoint, ContactMotion, TorqLine, PenEven, Wrench, PenHinge
from gtsam_custom_factors import WrenchInc, DispVar, EnergyElastic
from scipy.spatial.transform import Rotation as R
from gtsam.symbol_shorthand import G, N, O, C, T, U, S, W  # Symbols for Variables / G: Gripper, N: Object (rest), O: Object (equil), C: Contact, T: Command Rotation, U: Control Input, S: Grasp Parameters, W: Wrench
                                                           # O(int(1e5)+i): displacement from object resting pose to equilibrium pose in canonical coordinate (Rx, Ry, Rz, x, y, z)
from gtsam.symbol_shorthand import P, D, U, A, B, E, F, H, M, I, J, K, L, Q, Z, V, X, Y # Symbols for Factors
from collections import deque
########################################################

class gtsam_graph:

    def __init__(self, reach=62, normal_vec=np.array([0,0,1])):

        self.reach = reach # prior estimate of how far is the object bottom surface is from the gripper center
        self.normal_vec = normal_vec # surface normal of the ground
        self.touch = False # flag that indicates whether the object is touching the environment
        self.error_raise = False
        self.cf_thres_1 = 0.12 #.75 # threshold for detecting contact formulation transition from a point to line
        self.cf_thres_1_ = 0.08
        self.cf_thres_2 = 0.15 #.75 # threshold for detecting contact formulation transition from a line to patch
        self.cf_detect_on_1, self.cf_detect_on_2 = False, False
        
        ### NOISE MODELS FOR EACH FACTORS + SOME HYPERPARAMETERS ###
        # For 6-D noise models, each components are sigmas for [Rx, Ry, Rz, x, y, z]
        self.ALL_FIXED = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-8, 1e-8, 1e-8, 1e-6, 1e-6, 1e-6]))
        self.ALL_FIXED_ = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3]))
        self.ALL_FIXED__ = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2]))
        self.T_FIXED = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.inf, np.inf, np.inf, 1e-6, 1e-6, 1e-6]))
        self.T_FIXED_ = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.inf, np.inf, np.inf, 1e-3, 1e-3, 1e-3]))
        self.R_FIXED = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-8, 1e-8, 1e-8, np.inf, np.inf, np.inf]))
        self.R_FIXED_ = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e0, 1e0, 1e0, np.inf, np.inf, np.inf]))
        self.P_FIXED = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-7, 1e-7, 1e-7, np.inf, np.inf, 1e-5]))
        self.P_FIXED_ = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-7, 1e-7, np.inf, np.inf, np.inf, 1e-5]))
        self.H_FIXED = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.inf, np.inf, 1e-3, 1e-1, 1e-1, np.inf]))
        self.A_FIXED = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, np.inf, np.inf, np.inf, 1e-1]))
        self.RX_FIXED = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
                            gtsam.noiseModel.Diagonal.Sigmas(np.array([np.inf, 1e-3, 1e-4, 1e-3, 1e-3, 1e-3])))
        self.RX_FIXED__ = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
                            gtsam.noiseModel.Diagonal.Sigmas(np.array([np.inf, np.inf, np.inf, 1e-3, 1e-3, 1e-3])))
        self.RX_FIXED_ = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.inf, 1e-2, 1e-4, 1e-2, 1e-2, 1e-2]))
        self.rx_fixed = np.array([np.inf, 0, 1e-4, 1e-3, 1e-3, 1e-3])
        self.rx_fixed_clip = 1e-5
        self.RZ_FIXED = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-5, 1e-5, np.inf, 1e-3, 1e-3, 1e-3]))
        self.RYRZ_FIXED = gtsam.noiseModel.Diagonal.Sigmas(1e1*np.array([np.inf, 1e-1, 1e-1, np.inf, np.inf, np.inf]))
        self.sticky = np.array([np.inf, np.inf, np.inf, 2e-1, 2e-1, np.inf])
        self.sticky_ = 4*np.array([np.inf, np.inf, 4e-3, 2e-1, 2e-1, np.inf])
        self.STICK = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
                    gtsam.noiseModel.Diagonal.Sigmas(np.array([np.inf, np.inf, np.inf, 2e0, 2e0, np.inf])))
        self.STICK_ = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
                    gtsam.noiseModel.Diagonal.Sigmas(np.array([np.inf, np.inf, 4e-2, 2e0, 2e0, np.inf])))

        self.OBJECT_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e1]))
        self.CT_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3, 2e1, 2e1, 1e-1]))

        self.control_horizon = 10
        self.CONTROL_EFFORT = gtsam.noiseModel.Diagonal.Sigmas(1e1*np.array([8.7e-2, 8.7e-2, 8.7e-2, 3.5, 3.5, 3.5]))
        self.CONTROL_EFFORT_ = gtsam.noiseModel.Diagonal.Sigmas(1e1*np.array([8.7e-2, 8.7e-2, 2e-3, .1, .1, 1]))
        self.CONTROL_EFFORT__ = gtsam.noiseModel.Diagonal.Sigmas(2e0*np.array([8.7e-2, 8.7e-2, 2e-3, .1, .1, 1]))
        self.CONTROL_EFFORT___ = gtsam.noiseModel.Diagonal.Sigmas(1e1*np.array([8.7e-2, 8.7e-2, 2e-2, .1, 1, 1]))
        self.CONTROL_EFFORT____ = gtsam.noiseModel.Diagonal.Sigmas(1e1*np.array([8.7e-2, 8.7e-2, 8.7e-2, .1, .1, 3.5]))
        self.TRAJ_ERROR = gtsam.noiseModel.Diagonal.Sigmas(1e1*np.array([1e-1, 1e-1, 1e-1, 1e5, 1e5, 1e5]))
        self.TRAJ_ERROR_ = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
                            gtsam.noiseModel.Diagonal.Sigmas(1e1*np.array([1e-1, 1e-1, 1e-1, 1e5, 1e5, 1e5])))

        self.NO_ENERGY_FACTOR = False # If True, turns off the energy factor
        self.USE_RT_ESTIMATE = True # If True, use online estimate of grasp parameters
        self.OOCC_RELIEVE = True # relieves the geometric constraints right after the line transition, when the line estimate is not accurate
        self.EVEN_PENETRATION = True # Encourages even penetration during the line contact
        
        self.min_type = 'force'
        self.min_type_dict = {2: 'force'}

        # stiffness (grasp parameters): Stiffness in [Rx, Ry, Rz, x, y, z] (6-D) + CoC offset in [x, y, z] (3-D) ==> (9-D)
        self.STIFFNESS_PRIOR_ = np.array([5.5e-2,  2.1e-2,  5.3e-2,  16.7e-1, 11.8e-1,  7.1e-1,  0, 0, -5])
        self.STIFFNESS_PRIOR_[:6] = self.STIFFNESS_PRIOR_[:6]**-2
        self.STIFFNESS_NOISE_OFFLINE = gtsam.noiseModel.Diagonal.Sigmas(self.STIFFNESS_PRIOR_[:6]**-0.5)
        self.stiffness_prior_ = np.zeros(9)
        self.stiffness_prior_[0] = 5*self.STIFFNESS_PRIOR_[0]
        self.stiffness_prior_[1] = 0.5*self.STIFFNESS_PRIOR_[1]
        self.stiffness_prior_[2] = 1e-6
        self.stiffness_prior_[3:6] = np.clip(0.5*self.STIFFNESS_PRIOR_[3:6], 0.25*np.max(self.STIFFNESS_PRIOR_[3:6]), np.inf)
        self.stiffness_prior_[[6,7,8]] = np.array([0.5, 5, 10])
        self.STIFFNESS_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(self.stiffness_prior_)
        self.STIFFNESS_FIXED = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3,1e-3,1e-3,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]))

        # for wrench regresssion
        self.WRENCH_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([20, 20, 20, 2, 1, 1]))
        self.WRENCH_PREDICT_NOISE_WEAK = gtsam.noiseModel.Diagonal.Sigmas(np.array([5, 5, 5, 2.5e-1, 2.5e-1, 2.5e-1]))
        self.WRENCH_PREDICT_NOISE_STRONG = gtsam.noiseModel.Diagonal.Sigmas(1e-3*np.array([5, 5, 5, 2.5e-1, 2.5e-1, 2.5e-1]))

        # for torque balance at the extrinsic contact
        self.STIFFRATIO_NOISE = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
                                    gtsam.noiseModel.Diagonal.Sigmas(np.array([25, 25, 25, 2.5, np.inf, np.inf])))
        self.STIFFLINE_NOISE = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
                                    gtsam.noiseModel.Diagonal.Sigmas(np.array([25, 2.5])))
        
        self.MIN_FORCE = 5e0

        # noise model for tactile module measurements
        self.OU_sigma = 20*np.array([4e-4, 8e-4, 16e-4, 40e-3, 10e-3, 10e-3])
        self.OU_multiplier = 10*np.array([.1, .4, 1, 1, .1, .1])
        self.OU_clip_lower = np.array([.002, .002, .008, .4, .1, .1])
        self.OU_clip_upper = np.array([.008, .008, .008, .4, .4, .4])
        self.DEFORM_NOISE = gtsam.noiseModel.Diagonal.Sigmas(self.OU_sigma)

        # ros publishers and subscribers
        # these won't be used if just running replay
        if __name__ == "__main__":
            self.everything_pub = rospy.Publisher('/everything', Float64MultiArray, queue_size=1)
            self.stiffness_pub_Rx = rospy.Publisher('/stiffness/Rx', Float64, queue_size=10)
            self.stiffness_pub_Ry = rospy.Publisher('/stiffness/Ry', Float64, queue_size=10)
            self.stiffness_pub_Rz = rospy.Publisher('/stiffness/Rz', Float64, queue_size=10)
            self.stiffness_pub_x = rospy.Publisher('/stiffness/x', Float64, queue_size=10)
            self.stiffness_pub_y = rospy.Publisher('/stiffness/y', Float64, queue_size=10)
            self.stiffness_pub_z = rospy.Publisher('/stiffness/z', Float64, queue_size=10)
            self.offset_pub_x = rospy.Publisher('/offset/x', Float64, queue_size=10)
            self.offset_pub_y = rospy.Publisher('/offset/y', Float64, queue_size=10)
            self.offset_pub_z = rospy.Publisher('/offset/z', Float64, queue_size=10)
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
            self.force_x = rospy.Publisher('/force_est/x', Float64, queue_size=10)
            self.force_y = rospy.Publisher('/force_est/y', Float64, queue_size=10)
            self.force_z = rospy.Publisher('/force_est/z', Float64, queue_size=10)
            self.penetration_pub = rospy.Publisher('/penetration', Float64, queue_size=10)

            self.restart_sub = rospy.Subscriber("/restart_cart", Float64MultiArray, self.restart, queue_size=1)
            self.add_new_sub = rospy.Subscriber("/addnew_data", Float64MultiArray, self.add_new, queue_size=1)
            self.print_graph_sub = rospy.Subscriber("/print_graph", Float64, self.print_graph, queue_size=1)
            self.save_data_sub = rospy.Subscriber("/save_data", Float64, self.save_data, queue_size=1)
            self.cf_detection_1_sub = rospy.Subscriber("/cf_detection_1", Float64, self.cf_detection_callback_1, queue_size=1)
            self.cf_detection_2_sub = rospy.Subscriber("/cf_detection_2", Float64, self.cf_detection_callback_2, queue_size=1)

    def cf_detection_callback_1(self, data):
        self.cf_detect_on_1 = bool(data.data)
    def cf_detection_callback_2(self, data):
        self.cf_detect_on_2 = bool(data.data)

    def Pose_to_Matrix(self, p):
        m = np.zeros((4,4))
        m[:3,-1] = p.translation()
        m[:3,:3] = p.rotation().matrix()
        return m

    def publish_everything(self):
        waypoints = []
        for i in range(self.control_horizon):
            waypoints.append(self.Pose_to_Matrix(self.G0_world.compose(self.current_estimate.atPose3(G(self.i+i+1)))))
        everything = np.hstack((self.i, self.Pose_to_Matrix(self.ct_).flatten(), self.Pose_to_Matrix(self.obj_).flatten(),\
                                self.ct_cov.flatten(), self.force_est, self.wr_, self.disp_, self.st, np.array(waypoints).flatten(), self.mode, float(self.error_raise)))
        self.everything_pub.publish(Float64MultiArray(data=everything))
        self.stiffness_pub_Rx.publish(self.st[0]/np.pi*180)
        self.stiffness_pub_Ry.publish(self.st[1]/np.pi*180)
        self.stiffness_pub_Rz.publish(self.st[2]/np.pi*180)
        self.stiffness_pub_x.publish(self.st[3])
        self.stiffness_pub_y.publish(self.st[4])
        self.stiffness_pub_z.publish(self.st[5])
        self.offset_pub_x.publish(self.st[6])
        self.offset_pub_y.publish(self.st[7])
        self.offset_pub_z.publish(self.st[8])
        self.penetration_pub.publish(self.pen)
        self.deform_raw_Rx.publish(self.tact_raw[5])
        self.deform_raw_Ry.publish(self.tact_raw[4])
        self.deform_raw_Rz.publish(self.tact_raw[3])
        self.deform_raw_x.publish(self.tact_raw[0])
        self.deform_raw_y.publish(self.tact_raw[1])
        self.deform_raw_z.publish(self.tact_raw[2])
        self.deform_est_Rx.publish(self.tact_est[5])
        self.deform_est_Ry.publish(self.tact_est[4])
        self.deform_est_Rz.publish(self.tact_est[3])
        self.deform_est_x.publish(self.tact_est[0])
        self.deform_est_y.publish(self.tact_est[1])
        self.deform_est_z.publish(self.tact_est[2])
        self.force_x.publish(self.force_est[0])
        self.force_y.publish(self.force_est[1])
        self.force_z.publish(self.force_est[2])

    def push_back(self, graph, f, f_num, f_dict=None, f_id=None):
        ## adds a new factor to the graph and save the factor number in a dictionary ##
        graph.push_back(f)
        if f_dict != None and f_id != None:
            f_dict[f_id] = f_num[0]
        f_num[0] += 1

    def restart(self, data):
        ## Start/restart a graph
        if __name__ == "__main__":
            cart_init = np.asarray(data.data[:7])
            st_prior = np.asarray(data.data[7:16])
            self.reach = np.asarray(data.data[-4])
            self.NO_ENERGY_FACTOR = bool(data.data[-3])
            self.USE_RT_ESTIMATE = bool(data.data[-2])
            self.min_type = self.min_type_dict[data.data[-1]]
        else:
            cart_init = data
            st_prior = np.zeros(9)

        print("restart")
        print(f"cart_init: {cart_init}")
        print(f"NO_ENERGY_FACTOR: {self.NO_ENERGY_FACTOR}")
        print(f"USE_RT_ESTIMATE: {self.USE_RT_ESTIMATE}")
        print(f"min_type: {self.min_type}")

        if self.min_type == "force":
            self.minimum_nominal = 5e-1
            self.F_target = 5e-1
            self.epsilon = 1e-1

        self.cart_init = cart_init.copy()
        self.G0_world = gtsam.Pose3(gtsam.Rot3(R.from_quat(cart_init[3:]).as_matrix()), cart_init[:3])
        self.r_g_init = R.from_quat(self.cart_init[3:])
        self.tactile_buffer = deque(maxlen=10)
        self.tactile_buffer.append(np.zeros(6))
        self.gt = np.zeros(6)
        self.touch = False
        self.error_raise = False
        self.cf_detect_on_1, self.cf_detect_on_2 = False, False

        self.mode = 0 # point: 0 / line: 1 / patch: 2
        
        ## Since, we do not know the orientation of the object, we first just assume it's parallel to the ground (normal_vec).
        ## Under the assumpiton, the below line computes a relative rotation matrix from gripper to object bottom surface.
        R_g = self.r_g_init.as_matrix()
        v = np.cross(np.array([0, 0, 1]), np.linalg.inv(R_g) @ self.normal_vec)
        c = np.dot(np.array([0, 0, 1]), np.linalg.inv(R_g) @ self.normal_vec)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        self.R_go = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        ## Resets the graph
        self.reset_graph(st_prior)
        if __name__ == "__main__":
            self.publish_everything()

    def reset_graph(self, st_prior):

        self.G_0 = gtsam.Pose3()

        self.i = 0 # current timestep
        self.j = 0 # timestep at the end of the control horizon

        parameters = gtsam.ISAM2Params()
        #parameters.setOptimizationParams(gtsam.ISAM2DoglegParams()) # enable this to use Powell's Dogleg optimizer
        parameters.setRelinearizeThreshold(0.01)
        parameters.setRelinearizeSkip(1)

        self.isam = gtsam.ISAM2(parameters)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.f_num = [0]
        self.f_dict = {}

        ## Factors for the initial timestep ##
        self.push_back(self.graph, gtsam.PriorFactorPose3(G(0), self.G_0, self.ALL_FIXED), self.f_num, self.f_dict, P(0)) # F_gp
        self.push_back(self.graph, gtsam.BetweenFactorPose3(G(0), N(0),
                    gtsam.Pose3(gtsam.Rot3(self.R_go), self.R_go @ np.array([0, 0, -self.reach])), self.OBJECT_PRIOR_NOISE), self.f_num) # F_op
        self.push_back(self.graph, DispDiff(N(0), O(0), G(0), G(0),
                    gtsam.Pose3(), self.ALL_FIXED__, False), self.f_num, self.f_dict, B(0)) # F_tac
        self.push_back(self.graph, DispVar(G(0), N(0), O(0), O(int(1e5)), self.ALL_FIXED__), self.f_num)
        self.push_back(self.graph, gtsam.BetweenFactorPose3(O(0), D(0), gtsam.Pose3(gtsam.Rot3(), [0, 0, 0]), self.CT_PRIOR_NOISE), self.f_num)
        self.push_back(self.graph, gtsam.BetweenFactorPose3(C(0), D(0), gtsam.Pose3(gtsam.Rot3(), [0, 0, 0]), self.T_FIXED), self.f_num)
        self.push_back(self.graph, gtsam.PriorFactorPose3(C(0), gtsam.Pose3(self.R_go), self.R_FIXED), self.f_num)
        
        # F_kp
        if self.USE_RT_ESTIMATE:
            if np.all(st_prior == np.zeros(9)):
                self.push_back(self.graph, gtsam.PriorFactorVector(S(0), self.STIFFNESS_PRIOR_, self.STIFFNESS_PRIOR_NOISE), self.f_num, self.f_dict, 123456)
            else:
                self.push_back(self.graph, gtsam.PriorFactorVector(S(0), st_prior, self.STIFFNESS_PRIOR_NOISE), self.f_num, self.f_dict, 123456)
        else:
            if np.all(st_prior == np.zeros(9)):
                self.push_back(self.graph, gtsam.PriorFactorVector(S(0), self.STIFFNESS_PRIOR_, self.STIFFNESS_FIXED), self.f_num, self.f_dict, 123456)
            else:
                self.push_back(self.graph, gtsam.PriorFactorVector(S(0), st_prior, self.STIFFNESS_FIXED), self.f_num, self.f_dict, 123456)
        
        self.push_back(self.graph, Wrench(G(0), N(0), O(0), W(0), S(0), self.WRENCH_NOISE), self.f_num, self.f_dict, Q(int(1e5)))
        self.push_back(self.graph, gtsam.PriorFactorVector(W(0), np.zeros(6), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4]))), self.f_num)

        ## Initial guesses ##
        self.initial_estimate.insert(G(0), gtsam.Pose3())
        self.initial_estimate.insert(N(0), gtsam.Pose3(gtsam.Rot3(self.R_go), self.R_go@np.array([0, 0, -self.reach])))
        self.initial_estimate.insert(O(0), gtsam.Pose3(gtsam.Rot3(self.R_go), self.R_go@np.array([0, 0, -self.reach])))
        self.initial_estimate.insert(C(0), gtsam.Pose3(gtsam.Rot3(self.R_go), self.R_go@np.array([0, 0, -self.reach])))
        self.initial_estimate.insert(D(0), gtsam.Pose3(gtsam.Rot3(self.R_go), self.R_go@np.array([0, 0, -self.reach])))
        self.initial_estimate.insert(W(0), np.zeros(6))
        self.initial_estimate.insert(S(0), self.STIFFNESS_PRIOR_)
        self.initial_estimate.insert(O(int(1e5)), np.zeros(6))

        ## Update the graph ##
        self.isam.update(self.graph, self.initial_estimate)
        self.graph.resize(0)
        self.initial_estimate.clear()
        
        ## Get current estimate after the update ##
        self.current_estimate = self.isam.calculateEstimate()
        self.grp, self.grp_ = self.current_estimate.atPose3(G(0)), self.current_estimate.atPose3(G(0))
        self.nob, self.nob_ = self.current_estimate.atPose3(N(0)), self.current_estimate.atPose3(N(0))
        self.obj, self.obj_ = self.current_estimate.atPose3(O(0)), self.current_estimate.atPose3(O(0))
        self.ct, self.ct_ = self.current_estimate.atPose3(C(0)), self.current_estimate.atPose3(C(0))
        self.wr, self.wr_ = self.current_estimate.atVector(W(0)), self.current_estimate.atVector(W(0))
        self.disp, self.disp_ = self.current_estimate.atVector(O(int(1e5))), self.current_estimate.atVector(O(int(1e5)))
        self.st = self.current_estimate.atVector(S(0))
        self.pen = 0
        self.cf_signal = deque(maxlen=3)
        for _ in range(3): self.cf_signal.append(0)
        self.tact_raw = np.zeros(6)
        self.tact_est = np.zeros(6)
        self.force_est = np.zeros(3)

        ## Get covariance (confidence) of the estimate ##
        self.marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(),
                                         self.isam.calculateEstimate())
        self.ct_cov = self.marginals.marginalCovariance(C(self.i))
        self.obj_cov = self.marginals.marginalCovariance(O(self.i))

        # Computes the D-Optimality Criterion for Contact Estimation, which is just similar to the determinant of the covariance matrix
        d_opt = self.ct_cov[3:,3:].diagonal().prod()**(1/3) 

        ## Adds the initial control horizon ##
        for _ in range(self.control_horizon):

            self.j += 1

            # F_rot
            self.push_back(self.graph, gtsam.PriorFactorPose3(T(self.j), gtsam.Pose3(), self.TRAJ_ERROR), self.f_num, self.f_dict, H(self.j))
            self.push_back(self.graph, PoseDIff(G(0), G(self.j), T(self.j), self.ALL_FIXED_, False), self.f_num, self.f_dict, F(self.j))
            
            # F_motion
            self.push_back(self.graph, gtsam.PriorFactorPose3(U(self.j), gtsam.Pose3(), self.CONTROL_EFFORT____), self.f_num, self.f_dict, M(self.j))
            self.push_back(self.graph, ContactMotion(G(self.j-1), G(self.j), C(self.j-1), U(self.j), self.ALL_FIXED_, False), self.f_num, self.f_dict, E(self.j))
            
            self.push_back(self.graph, DispDiff(N(self.j-1), N(self.j), G(self.j-1), G(self.j), gtsam.Pose3(), self.ALL_FIXED_, False), self.f_num)

            self.push_back(self.graph, WrenchInc(O(int(1e5)+self.j-1), O(int(1e5)+self.j), W(self.j-1), W(self.j), S(0), self.WRENCH_PREDICT_NOISE_STRONG, False), self.f_num, self.f_dict, I(int(1e5)+self.j))
            self.push_back(self.graph, DispVar(G(self.j), N(self.j), O(self.j), O(int(1e5)+self.j), self.ALL_FIXED__), self.f_num)
            
            # F_E
            if not self.NO_ENERGY_FACTOR:
                self.push_back(self.graph, EnergyElastic(W(self.j), S(0), 10), self.f_num, self.f_dict, I(self.j))
            else:
                self.push_back(self.graph, EnergyElastic(W(self.j), S(0), 2000), self.f_num, self.f_dict, I(self.j))
            
            # F_pen
            if self.min_type == "force":
                self.minimum_threshold = np.hstack((np.linspace(0, self.minimum_nominal, self.control_horizon-4)[1:],np.array(5*[self.minimum_nominal])))
                self.push_back(self.graph, PenHinge(N(self.j), C(self.j), O(self.j), self.MIN_FORCE, self.minimum_threshold[self.j-1]), self.f_num, self.f_dict, J(self.j))
            else:
                raise NotImplementedError
            
            self.push_back(self.graph, DispDiff(O(self.j-1), O(self.j), C(self.j-1), C(self.j), gtsam.Pose3(), self.T_FIXED_, False), self.f_num)
            self.push_back(self.graph, gtsam.BetweenFactorPose3(C(self.j-1), C(self.j), gtsam.Pose3(), self.P_FIXED), self.f_num)
            self.push_back(self.graph, gtsam.BetweenFactorPose3(C(self.j-1), C(self.j), gtsam.Pose3(), self.STICK), self.f_num, self.f_dict, A(self.j))

            self.initial_estimate.insert(G(self.j), gtsam.Pose3())
            self.initial_estimate.insert(N(self.j), self.nob)
            self.initial_estimate.insert(O(self.j), self.obj)
            self.initial_estimate.insert(C(self.j), self.ct)
            self.initial_estimate.insert(U(self.j), gtsam.Pose3())
            self.initial_estimate.insert(T(self.j), gtsam.Pose3())
            self.initial_estimate.insert(W(self.j), self.wr)
            self.initial_estimate.insert(O(int(1e5)+self.j), self.disp)

            self.isam.update(self.graph, self.initial_estimate)
            self.isam.update()
            self.graph.resize(0)
            self.initial_estimate.clear()

            self.current_estimate = self.isam.calculateEstimate()
            self.grp = self.current_estimate.atPose3(G(self.j))
            self.nob = self.current_estimate.atPose3(N(self.j))
            self.obj = self.current_estimate.atPose3(O(self.j))
            self.ct = self.current_estimate.atPose3(C(self.j))
            self.wr = self.current_estimate.atVector(W(self.j))
            self.disp = self.current_estimate.atVector(O(int(1e5)+self.j))

        # waypoints (motion plan) are the sequence of gripper poses during the future control horizon
        self.waypoints = []
        for i in range(self.control_horizon):
            self.waypoints.append(self.current_estimate.atPose3(G(i+1)))

    def add_new(self, data):

        # Adds new measurements
        # data[:,:7]: cartesian coordinate (quaternion)
        # data[:,7:13]: tactile measurements (x, y, z, yaw(Rx), pitch(Ry), roll(Rx))
        # data[:,13:]: command rotation matrix
        if __name__ == "__main__":
            data_array = np.asarray(data.data).reshape(-1,7+6+9)
        else:
            data_array = np.hstack((data[0], data[1], data[2].reshape((-1,9))))

        cart_new_list, tactile_new_list, com_rot_list = [], [], []
        for i in range(data_array.shape[0]):
            cart_new_list.append(data_array[i,:7])
            tactile_new_list.append(data_array[i,7:13])
            com_rot_list.append(data_array[i,13:].reshape(3,3))

        remove_idx = [] # list of factors to be removed (or replaced)

        for k in range(len(cart_new_list)):
            cart_new = cart_new_list[k]
            tactile_new = tactile_new_list[k]
            com_rot = com_rot_list[k]
        
            self.cart = np.array(cart_new)
            self.tact_raw = tactile_new.copy()
            if self.touch:
                self.i += 1
                self.j += 1

            # touch detection
            if not self.touch:
                if np.linalg.norm(tactile_new / np.array([0.05, 0.05, 0.05, .2, .2, .2])) > 1.:
                    self.touch = True
                    print("TOUCH!")
                elif k != len(cart_new_list)-1:
                    continue
                else:
                    tactile_new[:] = 0

            if self.mode != 0:
                if self.i < self.it + 10:
                        com_rot = np.eye(3)

            # GRIPPER #
            xyz_world = self.cart[:3] - self.cart_init[:3]
            self.r_g = R.from_quat(self.cart[3:])
            xyz = self.r_g_init.inv().as_matrix().dot(xyz_world)
            ypr = (self.r_g_init.inv() * self.r_g).as_euler('zyx')
            self.gt = np.hstack((xyz, ypr))
            g_rot = R.from_euler('zyx', self.gt[3:]).as_matrix()
            g_trn = self.gt[:3]

            # TACTILE #
            TM_ = self.tactile_buffer[0].copy()
            TM_[3:] *= np.pi / 180
            TM_rot_ = R.from_euler('zyx', TM_[3:], degrees=False).as_matrix()
            TM_trn_ = TM_[:3].copy()
            TM = tactile_new.copy()
            TM[3:] *= np.pi / 180
            TM_rot = R.from_euler('zyx', TM[3:], degrees=False).as_matrix()
            TM_trn = TM[:3].copy()
            self.tactile_buffer.append(tactile_new.copy())
            if self.i == 0:
                self.tactile_buffer.popleft()

            self.DEFORM_NOISE_ = gtsam.noiseModel.Diagonal.Sigmas(self.OU_multiplier*np.clip(np.abs(np.hstack((R.from_matrix(TM_rot).as_rotvec(), TM_trn))), self.OU_clip_lower, self.OU_clip_upper))

            if self.i == 0:
                ## If the object is still not in touch, dump the old measurement (remove_idx) and update with the new measurment.
                remove_idx.append(self.f_dict[B(0)])
                remove_idx.append(self.f_dict[P(0)])

                # tactile deformation measurement (F_tac)
                self.push_back(self.graph, DispDiff(N(0), O(0), G(0), G(0),
                        gtsam.Pose3(gtsam.Rot3(TM_rot), TM_trn), self.ALL_FIXED__, False), self.f_num, self.f_dict, B(0))
                
                # gripper position (F_gp)
                self.push_back(self.graph, gtsam.PriorFactorPose3(G(0), gtsam.Pose3(gtsam.Rot3(g_rot), g_trn), self.ALL_FIXED_), self.f_num, self.f_dict, P(0))

            elif self.mode == 0: # point contact
                
                ## Remove some factors at the first control horizon timestep and add new measurement at timestep [self.i]
                remove_idx.append(self.f_dict[A(self.i)])
                remove_idx.append(self.f_dict[E(self.i)])
                remove_idx.append(self.f_dict[F(self.i)])
                remove_idx.append(self.f_dict[I(self.i)])
                remove_idx.append(self.f_dict[J(self.i)])
                if k == len(cart_new_list)-1:
                    for t in range(max(self.i+1,self.control_horizon+1),self.j-k):
                        remove_idx.append(self.f_dict[J(t)])
                remove_idx.append(self.f_dict[I(int(1e5)+self.i)])

                # F_cc
                # The strength of the stikcing factor is determined by the d_opt. 
                # In other words, if the current contact estimation is confident,
                # it is more likely to stick so the sticking factor is set stronger.
                d_opt = self.ct_cov[3:,3:].diagonal().prod()**(1/3)
                self.push_back(self.graph, gtsam.BetweenFactorPose3(C(self.i-1), C(self.i), gtsam.Pose3(), 
                    gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
                    gtsam.noiseModel.Diagonal.Sigmas(d_opt**0.5 / 2 * self.sticky))
                    ), self.f_num, self.f_dict, A(self.i))

                # gripper position (F_gp)
                self.push_back(self.graph, gtsam.PriorFactorPose3(G(self.i), gtsam.Pose3(gtsam.Rot3(g_rot), g_trn), self.ALL_FIXED_), self.f_num)

                # tactile deformation measurement (F_tac, F_tac_inc)
                self.push_back(self.graph, DispDiff(N(0), O(self.i), G(0), G(self.i),
                            gtsam.Pose3(gtsam.Rot3(TM_rot), TM_trn), self.DEFORM_NOISE_, False), self.f_num, self.f_dict, D(int(1e5)+self.i)) # F_tac
                self.push_back(self.graph, DispDiff(O(max(self.i-10,0)), O(self.i), G(max(self.i-10,0)), G(self.i),
                            gtsam.Pose3.between(gtsam.Pose3(gtsam.Rot3(TM_rot_), TM_trn_),
                                            gtsam.Pose3(gtsam.Rot3(TM_rot), TM_trn)), self.DEFORM_NOISE, False), self.f_num, self.f_dict, D(int(2e5)+self.i)) # F_tac_inc

                # This factor infers the wrench (F_wr, F_wr_inc) and grasp parameters (F_torq)
                self.push_back(self.graph, Wrench(G(self.i), N(self.i), O(self.i), W(self.i), S(0), self.WRENCH_NOISE), self.f_num, self.f_dict, Q(self.i+int(1e5))) # F_wr
                self.push_back(self.graph, TorqPoint(G(self.i), W(self.i), C(self.i), S(0), self.STIFFNESS_PRIOR_[:6], self.STIFFRATIO_NOISE), self.f_num, self.f_dict, Q(self.i)) # F_torq
                self.push_back(self.graph, WrenchInc(O(int(1e5)+self.i-1), O(int(1e5)+self.i), W(self.i-1), W(self.i), S(0), self.WRENCH_PREDICT_NOISE_WEAK, False), self.f_num, self.f_dict, I(int(1e5)+self.i)) # F_wr_inc

                ## Extend the control horizon one timestep further [self.j]

                self.push_back(self.graph, WrenchInc(O(int(1e5)+self.j-1), O(int(1e5)+self.j), W(self.j-1), W(self.j), S(0), self.WRENCH_PREDICT_NOISE_STRONG, False), self.f_num, self.f_dict, I(int(1e5)+self.j)) # F_wr_inc
                self.push_back(self.graph, DispVar(G(self.j), N(self.j), O(self.j), O(int(1e5)+self.j), self.ALL_FIXED__), self.f_num) 
                
                # command rotation at the end of control horizon (F_rot)
                self.push_back(self.graph, gtsam.PriorFactorPose3(T(self.j), gtsam.Pose3(gtsam.Rot3(com_rot), np.zeros(3)), self.TRAJ_ERROR), self.f_num, self.f_dict, H(self.j))
                self.push_back(self.graph, PoseDIff(G(0), G(self.j), T(self.j), self.ALL_FIXED_, False), self.f_num, self.f_dict, F(self.j))

                # Motion effort (F_motion)
                self.push_back(self.graph, gtsam.PriorFactorPose3(U(self.j), gtsam.Pose3(), self.CONTROL_EFFORT), self.f_num, self.f_dict, M(self.j))
                self.push_back(self.graph, ContactMotion(G(self.j-1), G(self.j), C(self.j-1), U(self.j), self.ALL_FIXED_, False), self.f_num, self.f_dict, E(self.j))

                # fixed grip
                self.push_back(self.graph, DispDiff(N(self.j-1), N(self.j), G(self.j-1), G(self.j), gtsam.Pose3(), self.ALL_FIXED_, False), self.f_num, self.f_dict, Z(self.j))

                # deformation energy (F_E)
                if not self.NO_ENERGY_FACTOR:
                    self.push_back(self.graph, EnergyElastic(W(self.j), S(0), 10), self.f_num, self.f_dict, I(self.j))
                else:
                    self.push_back(self.graph, EnergyElastic(W(self.j), S(0), 2000), self.f_num, self.f_dict, I(self.j)) # 2000 --> very weak
                
                # minimum penetration (F_pen)
                if self.min_type == "force":
                    if k == len(cart_new_list)-1:
                        # adjust target threshold to maintain consistent normal force
                        target_threshold = (self.epsilon*self.minimum_nominal + self.F_target*self.pen) / (self.epsilon + np.abs(self.force_est[2]))
                        target_threshold = np.clip(target_threshold, 0, 0.7)
                        self.minimum_threshold = np.hstack((np.linspace(self.minimum_threshold[min(k+1, self.control_horizon-1)], target_threshold, self.control_horizon-4)[1:],np.array(5*[target_threshold])))
                        for t in range(max(self.i+1,self.control_horizon+1),self.j+1):
                            self.push_back(self.graph, PenHinge(N(t), C(t), O(t), self.MIN_FORCE, self.minimum_threshold[t-self.j-1]), self.f_num, self.f_dict, J(t))
                else:
                    raise NotImplementedError
                
                # contact point's translational location (from the object perspective) must be fixed throughout time (F_oc)
                self.push_back(self.graph, DispDiff(O(self.j-1), O(self.j), C(self.j-1), C(self.j), gtsam.Pose3(), self.T_FIXED_, False), self.f_num, self.f_dict, V(self.j))

                # Pose rotation is redundant to represent the point, so fix the rotation to be constant.
                # Then, z-direction should also be fixed since we assume the ground to be flat. (F_cc)
                self.push_back(self.graph, gtsam.BetweenFactorPose3(C(self.j-1), C(self.j), gtsam.Pose3(), self.P_FIXED), self.f_num, self.f_dict, X(self.j))

                # If controlled well, it is less likely slip (while it could still slip a little bit).
                # Therefore, impose a weak (weaker than 'self.sticky') cost on horizontal contact point displacement.
                self.push_back(self.graph, gtsam.BetweenFactorPose3(C(self.j-1), C(self.j), gtsam.Pose3(), self.STICK), self.f_num, self.f_dict, A(self.j))

                self.initial_estimate.insert(G(self.j), self.grp)
                self.initial_estimate.insert(N(self.j), self.nob)
                self.initial_estimate.insert(O(self.j), self.obj)
                self.initial_estimate.insert(C(self.j), self.ct)
                self.initial_estimate.insert(U(self.j), gtsam.Pose3())
                self.initial_estimate.insert(T(self.j), gtsam.Pose3(gtsam.Rot3(com_rot), np.zeros(3)))
                self.initial_estimate.insert(W(self.j), self.wr)
                self.initial_estimate.insert(O(int(1e5)+self.j), self.disp)
            
            elif self.mode == 1 or self.mode == 2: # line
                ## Remove some factors at the first control horizon timestep and add new measurement at timestep [self.i]
                remove_idx.append(self.f_dict[A(self.i)])
                remove_idx.append(self.f_dict[E(self.i)])
                remove_idx.append(self.f_dict[F(self.i)])
                remove_idx.append(self.f_dict[I(self.i)])
                if k != len(cart_new_list)-1:
                    remove_idx.append(self.f_dict[J(self.i)])
                else:
                    for t in range(self.i,self.j-k):
                        remove_idx.append(self.f_dict[J(t)])
                remove_idx.append(self.f_dict[I(int(1e5)+self.i)])
                remove_idx.append(self.f_dict[V(self.i)])
                if self.i > self.it + 30 and V(self.i-30) in list(self.f_dict.keys()):
                    remove_idx.append(self.f_dict[V(self.i-30)])
                    self.push_back(self.graph, DispDiff(O(self.i-30-1), O(self.i-30), C(self.i-30-1), C(self.i-30), gtsam.Pose3(), self.RX_FIXED__, False), self.f_num, self.f_dict, V(self.i-30))
                if self.EVEN_PENETRATION: remove_idx.append(self.f_dict[Y(self.i)])

                # The strength of the stikcing factor is determined by the d_opt. 
                # In other words, if the current contact estimation is confident,
                # it is more likely to stick so the sticking factor is set stronger.
                # There is a difference from the point contact, that we also impose some cost on Rz (yaw) rotation,
                # because the object should not rotate perpendicular to the surface normal if it is sticking.
                d_opt_r = self.ct_cov[2,2]/(0.02**2)
                d_opt_t = self.ct_cov[3:,3:].diagonal().prod()**(1/3)
                d_opt = d_opt_r**(1/4) * d_opt_t**(3/4)
                self.push_back(self.graph, gtsam.BetweenFactorPose3(C(self.i-1), C(self.i), gtsam.Pose3(), 
                    gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
                    gtsam.noiseModel.Diagonal.Sigmas(d_opt**0.5 / 2 * self.sticky_))
                    ), self.f_num, self.f_dict, A(self.i))

                # gripper position
                self.push_back(self.graph, gtsam.PriorFactorPose3(G(self.i), gtsam.Pose3(gtsam.Rot3(g_rot), g_trn), self.ALL_FIXED_), self.f_num)

                # contact line/patch (from the object perspective) must be fixed throughout time
                if self.mode == 1: # line
                    if np.all(com_rot == np.eye(3)):
                        self.push_back(self.graph, DispDiff(O(self.i-1), O(self.i), C(self.i-1), C(self.i), gtsam.Pose3(), self.RX_FIXED__, False), self.f_num, self.f_dict, V(self.i))
                    else:
                        self.rx_fixed[1] = np.clip(1e-2 * self.ct_cov.diagonal()[2]**0.5, self.rx_fixed_clip, 1)
                        RX_FIXED = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
                                        gtsam.noiseModel.Diagonal.Sigmas(self.rx_fixed))
                        self.push_back(self.graph, DispDiff(O(self.i-1), O(self.i), C(self.i-1), C(self.i), gtsam.Pose3(), RX_FIXED, False), self.f_num, self.f_dict, V(self.i))
                elif self.mode == 2: # patch
                    self.push_back(self.graph, DispDiff(O(self.i-1), O(self.i), C(self.i-1), C(self.i), gtsam.Pose3(), self.ALL_FIXED__, False), self.f_num, self.f_dict, V(self.i))

                # tactile deformation measurement
                self.push_back(self.graph, DispDiff(N(self.it), O(self.i), G(self.it), G(self.i),
                            gtsam.Pose3(gtsam.Rot3(TM_rot), TM_trn), self.DEFORM_NOISE_, False), self.f_num, self.f_dict, D(int(1e5)+self.i))
                self.push_back(self.graph, DispDiff(O(max(self.i-10,self.it)), O(self.i), G(max(self.i-10,self.it)), G(self.i),
                            gtsam.Pose3.between(gtsam.Pose3(gtsam.Rot3(TM_rot_), TM_trn_),
                                            gtsam.Pose3(gtsam.Rot3(TM_rot), TM_trn)), self.DEFORM_NOISE, False), self.f_num, self.f_dict, D(int(2e5)+self.i))

                # This factor infers the stiffness ratio between different directions.
                # (similar to StiffnessRatioFactor in the point case)
                self.push_back(self.graph, Wrench(G(self.i), N(self.i), O(self.i), W(self.i), S(0), self.WRENCH_NOISE), self.f_num, self.f_dict, Q(self.i+int(1e5)))
                self.push_back(self.graph, WrenchInc(O(int(1e5)+self.i-1), O(int(1e5)+self.i), W(self.i-1), W(self.i), S(0), self.WRENCH_PREDICT_NOISE_WEAK, False), self.f_num, self.f_dict, I(int(1e5)+self.i))
                if self.mode == 1:
                    self.push_back(self.graph, TorqLine(G(self.i), W(self.i), C(self.i), S(0), self.STIFFNESS_PRIOR_[:6], self.STIFFLINE_NOISE), self.f_num, self.f_dict, Q(self.i))

                ## Extend the control horizon one timestep further [self.j]

                self.push_back(self.graph, WrenchInc(O(int(1e5)+self.j-1), O(int(1e5)+self.j), W(self.j-1), W(self.j), S(0), self.WRENCH_PREDICT_NOISE_STRONG, False), self.f_num, self.f_dict, I(int(1e5)+self.j))
                self.push_back(self.graph, DispVar(G(self.j), N(self.j), O(self.j), O(int(1e5)+self.j), self.ALL_FIXED__), self.f_num)

                # command rotation at the end of control horizon
                self.push_back(self.graph, gtsam.PriorFactorPose3(T(self.j), gtsam.Pose3(gtsam.Rot3(com_rot), np.zeros(3)), self.TRAJ_ERROR_), self.f_num, self.f_dict, H(self.j))
                self.push_back(self.graph, PoseDIff(G(self.it), G(self.j), T(self.j), self.ALL_FIXED_, False), self.f_num, self.f_dict, F(self.j))
                
                # Motion effort
                if self.ct_cov.diagonal()[2]**0.5 > 0.06:
                    self.push_back(self.graph, gtsam.PriorFactorPose3(U(self.j), gtsam.Pose3(), self.CONTROL_EFFORT_), self.f_num, self.f_dict, M(self.j))
                else:
                    self.push_back(self.graph, gtsam.PriorFactorPose3(U(self.j), gtsam.Pose3(), self.CONTROL_EFFORT___), self.f_num, self.f_dict, M(self.j))
                self.push_back(self.graph, ContactMotion(G(self.j-1), G(self.j), C(self.j-1), U(self.j), self.ALL_FIXED_, False), self.f_num, self.f_dict, E(self.j))
                
                # fixed grip
                self.push_back(self.graph, DispDiff(N(self.j-1), N(self.j), G(self.j-1), G(self.j), gtsam.Pose3(), self.ALL_FIXED_, False), self.f_num, self.f_dict, Z(self.j))
                
                # deformation energy
                if not self.NO_ENERGY_FACTOR:
                    self.push_back(self.graph, EnergyElastic(W(self.j), S(0), 10), self.f_num, self.f_dict, I(self.j))
                else:
                    self.push_back(self.graph, EnergyElastic(W(self.j), S(0), 2000), self.f_num, self.f_dict, I(self.j))
                
                # minimum penetration
                if self.min_type == "force":
                    if k == len(cart_new_list)-1:
                        target_threshold = (self.epsilon*self.minimum_nominal + self.F_target*self.pen) / (self.epsilon + np.abs(self.force_est[2]))
                        target_threshold = np.clip(target_threshold, 0, 0.7)
                        self.minimum_threshold = np.hstack((np.linspace(self.minimum_threshold[min(k+1, self.control_horizon-1)], target_threshold, self.control_horizon-4)[1:],np.array(5*[target_threshold])))
                        for t in range(self.i+1,self.j+1):
                            self.push_back(self.graph, PenHinge(N(t), C(t), O(t), self.MIN_FORCE, self.minimum_threshold[t-self.j-1]), self.f_num, self.f_dict, J(t))
                else:
                    raise NotImplementedError

                # even penetration along line/patch
                if self.EVEN_PENETRATION:
                    if self.mode == 1:
                        self.push_back(self.graph, PenEven(N(self.j), O(self.j), C(self.j), self.RYRZ_FIXED), self.f_num, self.f_dict, Y(self.j))
                    elif self.mode == 2:
                        self.push_back(self.graph, PenEven(N(self.j), O(self.j), C(self.j), self.R_FIXED_), self.f_num, self.f_dict, Y(self.j))

                # contact line/patch (from the object perspective) must be fixed throughout time
                # however in the planning future timesteps, if the contact estimate is inconfident,
                # it must be set weak.
                # If it's set strong, then the controller will only try to rotate around the current line estimate.
                if self.mode == 1:
                    if self.OOCC_RELIEVE:
                        self.rx_fixed[1] = np.clip(1 * self.ct_cov.diagonal()[2]**0.5, 0.01, 1)
                        RX_FIXED = gtsam.noiseModel.Diagonal.Sigmas(self.rx_fixed)
                        self.push_back(self.graph, DispDiff(O(self.j-1), O(self.j), C(self.j-1), C(self.j), gtsam.Pose3(), RX_FIXED, False), self.f_num, self.f_dict, V(self.j))
                    else:                
                        self.push_back(self.graph, DispDiff(O(self.j-1), O(self.j), C(self.j-1), C(self.j), gtsam.Pose3(), self.RX_FIXED_, False), self.f_num, self.f_dict, V(self.j))
                elif self.mode == 2:
                    self.push_back(self.graph, DispDiff(O(self.j-1), O(self.j), C(self.j-1), C(self.j), gtsam.Pose3(), self.ALL_FIXED__, False), self.f_num, self.f_dict, V(self.j))

                # The contact line always stays on the environment surface
                self.push_back(self.graph, gtsam.BetweenFactorPose3(C(self.j-1), C(self.j), gtsam.Pose3(), self.P_FIXED_), self.f_num, self.f_dict, X(self.j))

                # If controlled well, it is less likely slip (while it could still slip a little bit).
                # Therefore, impose a weak (weaker than 'self.sticky_') cost on horizontal contact point displacement.
                self.push_back(self.graph, gtsam.BetweenFactorPose3(C(self.j-1), C(self.j), gtsam.Pose3(), self.STICK_), self.f_num, self.f_dict, A(self.j))

                self.initial_estimate.insert(G(self.j), self.grp)
                self.initial_estimate.insert(N(self.j), self.nob)
                self.initial_estimate.insert(O(self.j), self.obj)
                self.initial_estimate.insert(C(self.j), self.ct)
                self.initial_estimate.insert(U(self.j), gtsam.Pose3())
                self.initial_estimate.insert(T(self.j), gtsam.Pose3(gtsam.Rot3(com_rot), np.zeros(3)))
                self.initial_estimate.insert(W(self.j), self.wr)
                self.initial_estimate.insert(O(int(1e5)+self.j), self.disp)

        # Computes F_torq before the update (this is not necessary, it's just for reference)
        d_ = self.disp_
        d = d_ + gtsam.Pose3.Logmap(gtsam.Pose3.between(gtsam.Pose3(gtsam.Rot3(TM_rot_), TM_trn_), gtsam.Pose3(gtsam.Rot3(TM_rot), TM_trn)))
        dc = np.array([d[0], d[1], d[2], d[3]+d[1]*self.st[8]-d[2]*self.st[7], d[4]+d[2]*self.st[6]-d[0]*self.st[8], d[5]+d[0]*self.st[7]-d[1]*self.st[6]])
        dc_ = np.array([d_[0], d_[1], d_[2], d_[3]+d_[1]*self.st[8]-d_[2]*self.st[7], d_[4]+d_[2]*self.st[6]-d_[0]*self.st[8], d_[5]+d_[0]*self.st[7]-d_[1]*self.st[6]])
        values = gtsam.Values()
        values.insert(G(self.i), gtsam.Pose3(gtsam.Rot3(g_rot), g_trn))
        values.insert(W(self.i), self.wr_ + self.st[:6]*(dc-dc_))
        values.insert(C(self.i), self.ct_)
        values.insert(S(0), self.st)
        factor = TorqPoint(G(self.i), W(self.i), C(self.i), S(0), self.STIFFNESS_PRIOR_[:6], self.STIFFRATIO_NOISE)
        self.sr_before_update = factor.whitenedError(values)[:3]

        try: 
            self.isam.update(self.graph, self.initial_estimate, gtsam.KeyVector(remove_idx))
            self.current_estimate = self.isam.calculateEstimate()
        except RuntimeError:
            print("Runtime Error Occurred, Restarting ISAM Graph")
            self.error_raise = True
        self.graph.resize(0)
        self.initial_estimate.clear()

        if Z(self.i) in list(self.f_dict.keys()):
            if np.linalg.norm(self.isam.getFactorsUnsafe().at(self.f_dict[Z(self.i)]).whitenedError(self.current_estimate)) > 10:
                print("error raised, too large cost term")
                self.error_raise = True
        
        self.grp, self.grp_ = self.current_estimate.atPose3(G(self.j)), self.current_estimate.atPose3(G(self.i))
        self.nob, self.nob_ = self.current_estimate.atPose3(N(self.j)), self.current_estimate.atPose3(N(self.i))
        self.obj, self.obj_ = self.current_estimate.atPose3(O(self.j)), self.current_estimate.atPose3(O(self.i))
        self.ct, self.ct_ = self.current_estimate.atPose3(C(self.j)), self.current_estimate.atPose3(C(self.i))
        self.wr, self.wr_ = self.current_estimate.atVector(W(self.j)), self.current_estimate.atVector(W(self.i))
        self.disp, self.disp_ = self.current_estimate.atVector(O(int(1e5)+self.j)), self.current_estimate.atVector(O(int(1e5)+self.i))
        self.st = self.current_estimate.atVector(S(0))
        self.pen = - self.ct_.inverse().compose(self.nob_.compose(self.obj_.inverse().compose(self.ct_))).translation()[2]
        deform = (self.nob_.inverse().compose(self.grp_)).inverse().compose(self.obj_.inverse().compose(self.grp_))
        deform_trn = deform.translation()
        deform_rot = R.from_matrix(deform.rotation().matrix()).as_euler('zyx', True)
        self.tact_est = np.hstack((deform_trn, deform_rot))
        self.force_est = self.r_g_init.as_matrix() @ self.grp_.rotation().matrix() @ self.wr_[3:]

        self.marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(),
                                         self.isam.calculateEstimate())
        self.ct_cov = self.marginals.marginalCovariance(C(self.i))
        self.obj_cov = self.marginals.marginalCovariance(O(self.i))

        self.d_opt = self.ct_cov[3:,3:].diagonal().prod()**(1/3)

        self.u = self.grp_.inverse().compose(self.ct_).rotation().rotate(np.array([0,0,1]))
        self.s_trn_c_z_squared = np.linalg.norm(self.u * self.st[3:6]**-0.5)**2
        self.v = np.cross(self.grp_.inverse().compose(self.ct_).translation(), self.u)
        self.s_rot_eff_sn = np.linalg.norm(self.v * self.st[:3]**-0.5)**2
        s_total_eff_sn = self.s_trn_c_z_squared + self.s_rot_eff_sn / np.clip(self.d_opt,2,np.inf)
        self.pen_dist_thres = s_total_eff_sn * self.minimum_threshold

        try:
            if self.mode == 0:
                self.cf_signal_ = np.linalg.norm(self.isam.getFactorsUnsafe().at(self.f_dict[Q(self.i)]).whitenedError(self.current_estimate)[:3])
            else:
                self.cf_signal_ = np.linalg.norm(self.isam.getFactorsUnsafe().at(self.f_dict[Q(self.i)]).whitenedError(self.current_estimate)[:1])
        except:
            self.cf_signal_ = 0

        ## Detect Contract Formation Transition
        if self.mode == 0 and self.cf_detect_on_1:
            self.cf_signal.append(np.linalg.norm(self.isam.getFactorsUnsafe().at(self.f_dict[Q(self.i)]).whitenedError(self.current_estimate)[:3]))
            print(f"{self.cf_signal[-1]}, {np.mean(self.cf_signal)}")
            if self.cf_signal[-1] > self.cf_thres_1 or np.mean(self.cf_signal) > self.cf_thres_1_:
                self.mode = 1
                print("CF transition (point -> line) detected!")
                self.tactile_buffer.clear()
                self.tactile_buffer.append(tactile_new.copy())
                if __name__ == "__main__": self.publish_everything()
                self.cf_transition()
        elif self.mode == 1 and self.cf_detect_on_2:
            self.cf_signal.append(np.linalg.norm(self.isam.getFactorsUnsafe().at(self.f_dict[Q(self.i)]).whitenedError(self.current_estimate)[:1]))
            if self.cf_signal[-1] > self.cf_thres_2:
                self.mode = 2
                print("CF transition (line -> patch) detected!")
                self.cf_transition()

        if __name__ == "__main__":
            self.publish_everything()

    def cf_transition(self):
        ## Transfer the factor graph from one contact formulation to another by removing, modifying, and adding some factors
        self.i += 1
        self.j += 1
        self.it = self.i

        remove_idx = []
        remove_idx.append(self.f_dict[E(self.i)]) # CONTACT MOTION
        remove_idx.append(self.f_dict[F(self.i)]) # GRIPPER TRAJ
        remove_idx.append(self.f_dict[I(self.i)]) # DEFORM ENERGY
        remove_idx.append(self.f_dict[J(self.i)]) # MIN PENETRATION
        remove_idx.append(self.f_dict[Z(self.i)]) # NNGG
        remove_idx.append(self.f_dict[V(self.i)]) # OOCC
        remove_idx.append(self.f_dict[X(self.i)]) # CC
        remove_idx.append(self.f_dict[A(self.i)])
        remove_idx.append(self.f_dict[I(int(1e5)+self.i)])
        if self.mode == 2:
            remove_idx.append(self.f_dict[Y(self.i)]) # NOC ALIGN
        for i in range(1,16):
            remove_idx.append(self.f_dict[Q(self.i-i)])

        # After the transition, now we know that the contact line or patch lie on the ground (normal_vec).
        # Therefore, we assume the object bottom surface is parallel to the ground.
        # If it's in the line contact mode, the true object bottom is still not known.
        R_g = self.r_g.as_matrix()
        z_g = R_g[:,-1]
        v = np.cross(z_g, self.normal_vec)
        c = np.dot(z_g, self.normal_vec)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        go_rot = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        # Update the gripper-object relative pose based on the above assumption.
        self.push_back(self.graph, gtsam.BetweenFactorPose3(G(self.i), O(self.i), gtsam.Pose3(gtsam.Rot3(go_rot), np.zeros(3)), self.H_FIXED), self.f_num)

        # Gripper pose remains the same
        self.push_back(self.graph, gtsam.BetweenFactorPose3(G(self.i-1), G(self.i), gtsam.Pose3(), self.ALL_FIXED_), self.f_num)

        # Although the gripper-object pose is changed, the relative deformation should remain the same.
        self.push_back(self.graph, gtsam.BetweenFactorVector(O(int(1e5)+self.i-1), O(int(1e5)+self.i), np.zeros(6), self.ALL_FIXED_), self.f_num)
        self.push_back(self.graph, WrenchInc(O(int(1e5)+self.i-1), O(int(1e5)+self.i), W(self.i-1), W(self.i), S(0), self.WRENCH_PREDICT_NOISE_WEAK, False), self.f_num, self.f_dict, I(int(1e5)+self.i))

        # The contact line should lie on the object bottom surface
        self.push_back(self.graph, gtsam.BetweenFactorPose3(O(self.i), C(self.i), gtsam.Pose3(), self.A_FIXED), self.f_num)
        if self.mode == 1: # point -> line
            # in the point contact mode we fixed the rotation of the contact,
            # but in the line contact, we should relieve the constrain on Rz (yaw) direction rotation.
            self.push_back(self.graph, gtsam.BetweenFactorPose3(C(self.i-1), C(self.i), gtsam.Pose3(), self.RZ_FIXED), self.f_num)
            self.push_back(self.graph, TorqLine(G(self.i), W(self.i), C(self.i), S(0), self.STIFFNESS_PRIOR_[:6], self.STIFFLINE_NOISE), self.f_num, self.f_dict, Q(self.i))
        elif self.mode == 2: # line -> patch
            self.push_back(self.graph, gtsam.BetweenFactorPose3(C(self.i-1), C(self.i), gtsam.Pose3(), self.ALL_FIXED_), self.f_num)            
        
        for t in range(self.i+1, self.j):
            remove_idx.append(self.f_dict[M(t)]) # CONTROL EFFORT --> Later: Shrink Rz direction
            remove_idx.append(self.f_dict[H(t)]) # GRIPPER TRAJ ERROR COST --> Perpendicular to the current contact line
            remove_idx.append(self.f_dict[F(t)]) # GRIPPER TRAJ --> set the reference as the transition
            remove_idx.append(self.f_dict[V(t)]) # line: OOCC --> [inf, inf, inf, 0, 0, 0] => [inf, 0, 0, 0, 0, 0]
                                                 # patch: OOCC --> [inf, 0, 0, 0, 0, 0] => [0, 0, 0, 0, 0, 0]
            if self.mode == 1:
                remove_idx.append(self.f_dict[X(t)]) # CC --> [0, 0, 0, 1, 1, 0] => [0, 0, 1, 1, 1, 0]
                remove_idx.append(self.f_dict[A(t)])
        
        # Modify the factors in the control horizon timesteps
        for t in range(self.i+1, self.j+1):
            self.push_back(self.graph, gtsam.PriorFactorPose3(U(t), gtsam.Pose3(), self.CONTROL_EFFORT__), self.f_num, self.f_dict, M(t))
            self.push_back(self.graph, gtsam.PriorFactorPose3(T(t), gtsam.Pose3(), self.TRAJ_ERROR_), self.f_num, self.f_dict, H(t))
            self.push_back(self.graph, PoseDIff(G(self.it), G(t), T(t), self.ALL_FIXED_, False), self.f_num, self.f_dict, F(t))

            if self.mode == 1:
                if self.OOCC_RELIEVE:
                    self.rx_fixed[1] = 1
                    RX_FIXED = gtsam.noiseModel.Diagonal.Sigmas(self.rx_fixed)
                    self.push_back(self.graph, DispDiff(O(t-1), O(t), C(t-1), C(t), gtsam.Pose3(), RX_FIXED, False), self.f_num, self.f_dict, V(t))
                else:                
                    self.push_back(self.graph, DispDiff(O(t-1), O(t), C(t-1), C(t), gtsam.Pose3(), self.RX_FIXED_, False), self.f_num, self.f_dict, V(t))

                self.push_back(self.graph, gtsam.BetweenFactorPose3(C(t-1), C(t), gtsam.Pose3(), self.P_FIXED_), self.f_num, self.f_dict, X(t))
                self.push_back(self.graph, gtsam.BetweenFactorPose3(C(t-1), C(t), gtsam.Pose3(), self.STICK_), self.f_num, self.f_dict, A(t))
                if self.EVEN_PENETRATION:
                    self.push_back(self.graph, PenEven(N(t), O(t), C(t), self.RYRZ_FIXED), self.f_num, self.f_dict, Y(t))
            elif self.mode == 2:
                self.push_back(self.graph, DispDiff(O(t-1), O(t), C(t-1), C(t), gtsam.Pose3(), self.ALL_FIXED__, False), self.f_num, self.f_dict, V(t))
                if self.EVEN_PENETRATION:
                    self.push_back(self.graph, PenEven(N(t), O(t), C(t), self.R_FIXED_), self.f_num, self.f_dict, Y(t))
                if t == self.j:
                    self.push_back(self.graph, gtsam.BetweenFactorPose3(C(t-1), C(t), gtsam.Pose3(), self.P_FIXED_), self.f_num, self.f_dict, X(t))
                    self.push_back(self.graph, gtsam.BetweenFactorPose3(C(t-1), C(t), gtsam.Pose3(), self.STICK_), self.f_num, self.f_dict, A(t))
        
        self.push_back(self.graph, WrenchInc(O(int(1e5)+self.j-1), O(int(1e5)+self.j), W(self.j-1), W(self.j), S(0), self.WRENCH_PREDICT_NOISE_STRONG, False), self.f_num, self.f_dict, I(int(1e5)+self.j))
        self.push_back(self.graph, DispVar(G(self.j), N(self.j), O(self.j), O(int(1e5)+self.j), self.ALL_FIXED__), self.f_num)
        self.push_back(self.graph, gtsam.PriorFactorPose3(U(self.j), gtsam.Pose3(), self.CONTROL_EFFORT__), self.f_num, self.f_dict, M(self.j))
        self.push_back(self.graph, ContactMotion(G(self.j-1), G(self.j), C(self.j-1), U(self.j), self.ALL_FIXED_, False), self.f_num, self.f_dict, E(self.j))
        self.push_back(self.graph, DispDiff(N(self.j-1), N(self.j), G(self.j-1), G(self.j), gtsam.Pose3(), self.ALL_FIXED_, False), self.f_num, self.f_dict, Z(self.j))
        self.push_back(self.graph, EnergyElastic(W(self.j), S(0), 10), self.f_num, self.f_dict, I(self.j))
        self.push_back(self.graph, PenHinge(N(self.j), C(self.j), O(self.j), self.MIN_FORCE, self.minimum_threshold[-1]), self.f_num, self.f_dict, J(self.j))

        self.initial_estimate.insert(G(self.j), self.grp)
        self.initial_estimate.insert(N(self.j), self.nob)
        self.initial_estimate.insert(O(self.j), self.obj)
        self.initial_estimate.insert(C(self.j), self.ct)
        self.initial_estimate.insert(U(self.j), gtsam.Pose3())
        self.initial_estimate.insert(T(self.j), gtsam.Pose3())
        self.initial_estimate.insert(W(self.j), self.wr)
        self.initial_estimate.insert(O(int(1e5)+self.j), self.disp)
        
        self.isam.update(self.graph, self.initial_estimate, gtsam.KeyVector(remove_idx))
        self.isam.update()
        self.graph.resize(0)
        self.initial_estimate.clear()

        self.current_estimate = self.isam.calculateEstimate()
        self.grp, self.grp_ = self.current_estimate.atPose3(G(self.j)), self.current_estimate.atPose3(G(self.i))
        self.nob, self.nob_ = self.current_estimate.atPose3(N(self.j)), self.current_estimate.atPose3(N(self.i))
        self.obj, self.obj_ = self.current_estimate.atPose3(O(self.j)), self.current_estimate.atPose3(O(self.i))
        self.ct, self.ct_ = self.current_estimate.atPose3(C(self.j)), self.current_estimate.atPose3(C(self.i))
        self.wr, self.wr_ = self.current_estimate.atVector(W(self.j)), self.current_estimate.atVector(W(self.i))
        self.disp, self.disp_ = self.current_estimate.atVector(O(int(1e5)+self.j)), self.current_estimate.atVector(O(int(1e5)+self.i))
        self.st = self.current_estimate.atVector(S(0))
        self.pen = - self.ct_.inverse().compose(self.nob_.compose(self.obj_.inverse().compose(self.ct_))).translation()[2]
        deform = (self.nob_.inverse().compose(self.grp_)).inverse().compose(self.obj_.inverse().compose(self.grp_))
        deform_trn = deform.translation()
        deform_rot = R.from_matrix(deform.rotation().matrix()).as_euler('zyx', True)
        self.tact_est = np.hstack((deform_trn, deform_rot))
        self.force_est = self.r_g_init.as_matrix() @ self.grp_.rotation().matrix() @ self.wr_[3:]

        self.marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(),
                                         self.isam.calculateEstimate())
        self.ct_cov = self.marginals.marginalCovariance(C(self.i))
        self.obj_cov = self.marginals.marginalCovariance(O(self.i))

    def return_all_cost(self, i):
        costs = []
        #              0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,  ,9   ,0,  ,1
        for symbol in [P,D,U,A,B,E,F,H,M,I,J,K,L,Q,Z,V,X,Y,'I_','Q_','D_','D__']:
            try:
                if symbol == 'I_':
                    cost = np.linalg.norm(self.isam.getFactorsUnsafe().at(self.f_dict[I(int(1e5)+i)]).whitenedError(self.current_estimate))
                elif symbol == 'Q_':
                    cost = np.linalg.norm(self.isam.getFactorsUnsafe().at(self.f_dict[Q(int(1e5)+i)]).whitenedError(self.current_estimate))
                elif symbol == 'D_':
                    cost = np.linalg.norm(self.isam.getFactorsUnsafe().at(self.f_dict[D(int(1e5)+i)]).whitenedError(self.current_estimate))
                elif symbol == 'D__':
                    cost = np.linalg.norm(self.isam.getFactorsUnsafe().at(self.f_dict[D(int(2e5)+i)]).whitenedError(self.current_estimate))
                else:
                    cost = np.linalg.norm(self.isam.getFactorsUnsafe().at(self.f_dict[symbol(i)]).whitenedError(self.current_estimate))
            except:
                cost = 0
            costs.append(cost)
        return costs

    def Logmap(self, pose, degree=True):
        logmap = gtsam.Pose3.Logmap(pose)
        if degree:
            logmap[:3] *= 180 / np.pi
        return logmap

    def get_Jacobian(self, f_shorthand, f_num, use_current_estimate=True, values=None):
        f = self.isam.getFactorsUnsafe().at(self.f_dict[f_shorthand(f_num)])
        if use_current_estimate:
            A, b = f.linearize(self.current_estimate).jacobianUnweighted() # A is Augmented Jacobian Matrix / b is Whitened Error
        else:
            v = gtsam.Values()
            for value in values:
                v.insert(value["v_id"], value["val"])
            A, b = f.linearize(v).jacobianUnweighted()

    def print_graph(self):
        print(self.isam.getFactorsUnsafe())

def main():
    print("start gtsam graph")
    rospy.init_node('gtsam_module', anonymous=True)
    while not rospy.is_shutdown():
        #gg = gtsam_graph(min_type="force", minimum_threshold=2e-2)
        gg = gtsam_graph()
        rospy.spin()

if __name__ == "__main__":
    import rospy
    from std_msgs.msg import Float64MultiArray, Int32, Bool, Float64
    main()