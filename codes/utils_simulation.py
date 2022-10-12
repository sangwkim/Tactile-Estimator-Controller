import numpy as np
import gtsam
from gtsam_custom_factors import DispDiff, Friction
from scipy.spatial.transform import Rotation as R
from gtsam.symbol_shorthand import G, N, O, C
from gtsam.symbol_shorthand import A, M, P, T
from collections import deque
########################################################

class graph_simulator:

    def __init__(self,
                 STIFFNESS = np.array([5.5e-2,  2.1e-2,  5.3e-2,  16.7e-1, 11.8e-1,  7.1e-1]),
                 MU = 0.15,
                 OU_sigma = np.array([1e-3, 1e-3, 1e-3, 2e-1, 5e-2, 5e-2]),
                 OU_theta = 0.1,
                 normal_vec = np.array([0,0,1])):

        self.STIFFNESS = STIFFNESS
        self.K = self.STIFFNESS**-2
        self.MU = MU
        self.normal_vec = normal_vec

        self.OU_sigma = OU_sigma
        self.OU_theta = OU_theta

        self.ALL_FIXED = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-8, 1e-8, 1e-8, 1e-6, 1e-6, 1e-6]))
        self.T_FIXED = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.inf, np.inf, np.inf, 1e-6, 1e-6, 1e-6]))
        self.R_FIXED = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-8, 1e-8, 1e-8, np.inf, np.inf, np.inf]))
        self.P_FIXED = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-8, 1e-8, 1e-8, np.inf, np.inf, 1e-8]))
        self.STICK_WEAK = 2e0
        self.STICK_STRONG = 1e-3
        
        self.STIFFNESS_NOISE = gtsam.noiseModel.Diagonal.Sigmas(self.STIFFNESS)

        self.FCONE_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3]))
        
        self.OU_noise = np.zeros(6)

    def push_back(self, graph, f, f_num, f_dict=None, f_id=None):
        ## adds a new factor to the graph and save the factor number in a dictionary ##
        graph.push_back(f)
        if f_dict != None and f_id != None:
            f_dict[f_id] = f_num[0]
        f_num[0] += 1

    def restart(self, cart_init, grasp_pose, height, contact_point):
        ## Start/restart a graph
        self.cart_init = cart_init.copy()
        self.grasp_pose = grasp_pose
        self.height = height
        self.contact_point = contact_point

        self.G0_world = gtsam.Pose3(gtsam.Rot3(R.from_quat(cart_init[3:]).as_matrix()), cart_init[:3])
        self.r_g_init = R.from_quat(self.cart_init[3:])
        self.gt = np.zeros(6)

        self.T_og = np.eye(4)
        self.T_og[:3,:3] = R.from_euler('zyx', self.grasp_pose[3:]).as_matrix()
        self.T_og[:3,-1] = np.array([0,0,self.height]) + self.grasp_pose[:3]

        self.touch = False

        if np.all(self.cart_init == np.array([0,0,0,0,0,0,1])):
            self.R_go = np.eye(3)
        else:
            R_g = self.r_g_init.as_matrix()
            v = np.cross(np.array([0, 0, 1]), np.linalg.inv(R_g) @ self.normal_vec)
            c = np.dot(np.array([0, 0, 1]), np.linalg.inv(R_g) @ self.normal_vec)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            self.R_go = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2)) # orientation of ground relative to gripper (init)

        ## Resets the graph
        self.reset_graph()

    def reset_graph(self):

        self.G_0 = gtsam.Pose3()

        self.i = 0 # current timestep

        parameters = gtsam.ISAM2Params()
        parameters.setOptimizationParams(gtsam.ISAM2DoglegParams()) # enable this to use Powell's Dogleg optimizer
        parameters.setRelinearizeThreshold(0.01)
        parameters.setRelinearizeSkip(1)

        self.isam = gtsam.ISAM2(parameters)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.f_num = [0]
        self.f_dict = {}

        ## Factors for the initial timestep ##
        self.push_back(self.graph, gtsam.PriorFactorPose3(G(0), self.G_0, self.ALL_FIXED), self.f_num, self.f_dict, P(0))
        self.push_back(self.graph, gtsam.BetweenFactorPose3(N(0), G(0),
                    gtsam.Pose3(gtsam.Rot3(self.T_og[:3,:3]), self.T_og[:3,-1]), self.ALL_FIXED), self.f_num)
        self.push_back(self.graph, gtsam.BetweenFactorPose3(N(0), O(0), gtsam.Pose3(), self.ALL_FIXED), self.f_num)
        self.push_back(self.graph, gtsam.BetweenFactorPose3(O(0), C(0), gtsam.Pose3(gtsam.Rot3(), self.contact_point), self.T_FIXED), self.f_num)
        self.push_back(self.graph, gtsam.PriorFactorPose3(C(0), gtsam.Pose3(self.R_go), self.R_FIXED), self.f_num)

        ## Initial guesses ##
        self.initial_estimate.insert(G(0), gtsam.Pose3())
        self.initial_estimate.insert(N(0), gtsam.Pose3(gtsam.Rot3(), np.array([0, 0, -self.height])))
        self.initial_estimate.insert(O(0), gtsam.Pose3(gtsam.Rot3(), np.array([0, 0, -self.height])))
        self.initial_estimate.insert(C(0), gtsam.Pose3(gtsam.Rot3(), np.array([0, 0, -self.height])))

        ## Update the graph ##
        self.isam.update(self.graph, self.initial_estimate)
        self.graph.resize(0)
        self.initial_estimate.clear()
        
        ## Get current estimate after the update ##
        self.current_estimate = self.isam.calculateEstimate()
        self.grp = self.current_estimate.atPose3(G(0))
        self.nob = self.current_estimate.atPose3(N(0))
        self.obj = self.current_estimate.atPose3(O(0))
        self.ct = self.current_estimate.atPose3(C(0))
        self.obj_0 = self.current_estimate.atPose3(O(0))
        self.grp_0 = self.current_estimate.atPose3(G(0))

    def add_new(self, cart_new):
        # Adds new measurements

        self.cart = cart_new.copy()

        # GRIPPER #
        xyz_world = self.cart[:3] - self.cart_init[:3]
        self.r_g = R.from_quat(self.cart[3:])
        xyz = self.r_g_init.inv().as_matrix().dot(xyz_world)
        ypr = (self.r_g_init.inv() * self.r_g).as_euler('zyx')
        self.gt = np.hstack((xyz, ypr))
        g_rot = R.from_euler('zyx', self.gt[3:]).as_matrix()
        g_trn = self.gt[:3]

        self.i += 1

        remove_idx = []
        if A(self.i-1) in list(self.f_dict.keys()): remove_idx.append(self.f_dict[A(self.i-1)])
        if M(self.i-1) in list(self.f_dict.keys()): remove_idx.append(self.f_dict[M(self.i-1)])
        if T(self.i-1) in list(self.f_dict.keys()): remove_idx.append(self.f_dict[T(self.i-1)])

        self.push_back(self.graph, DispDiff(O(0), O(self.i-1), G(0), G(self.i-1),
            (self.obj_0.inverse().compose(self.grp_0)).inverse().compose(self.obj.inverse().compose(self.grp)), self.ALL_FIXED, False), self.f_num)
    
        ####### SIMULATE ONE STEP #######
        self.push_back(self.graph, gtsam.PriorFactorPose3(G(self.i), gtsam.Pose3(gtsam.Rot3(g_rot), g_trn), self.ALL_FIXED), self.f_num)
        self.push_back(self.graph, DispDiff(N(self.i-1), N(self.i), G(self.i-1), G(self.i), gtsam.Pose3(), self.ALL_FIXED, False), self.f_num)
        
        self.push_back(self.graph, DispDiff(O(0), O(self.i), G(0), G(self.i), gtsam.Pose3(), self.STIFFNESS_NOISE, False), self.f_num, self.f_dict, A(self.i))

        self.push_back(self.graph, DispDiff(O(self.i-1), O(self.i), C(self.i-1), C(self.i), gtsam.Pose3(), self.T_FIXED, False), self.f_num)

        if not self.touch:
            self.push_back(self.graph, gtsam.BetweenFactorPose3(C(self.i-1), C(self.i), gtsam.Pose3(), self.R_FIXED), self.f_num)
        else:
            self.push_back(self.graph, gtsam.BetweenFactorPose3(C(self.i-1), C(self.i), gtsam.Pose3(), self.P_FIXED), self.f_num)
            self.push_back(self.graph, Friction(G(self.i-1), N(self.i-1), O(self.i-1), C(self.i-1), C(self.i), self.MU, self.K, self.STICK_WEAK, self.STICK_STRONG), self.f_num, self.f_dict, M(self.i))

        self.initial_estimate.insert(G(self.i), self.grp)
        self.initial_estimate.insert(N(self.i), self.nob)
        self.initial_estimate.insert(O(self.i), self.obj)
        self.initial_estimate.insert(C(self.i), self.ct)

        self.isam.update(self.graph, self.initial_estimate, gtsam.KeyVector(remove_idx))
        self.isam.update()
        self.graph.resize(0)
        self.initial_estimate.clear()

        self.current_estimate = self.isam.calculateEstimate()
        self.grp = self.current_estimate.atPose3(G(self.i))
        self.nob = self.current_estimate.atPose3(N(self.i))
        self.obj = self.current_estimate.atPose3(O(self.i))
        self.ct = self.current_estimate.atPose3(C(self.i))

        self.disp = gtsam.Pose3.Logmap( ( self.nob.inverse() * self.grp ).inverse() * ( self.obj.inverse() * self.grp ) )
        self.force = self.r_g_init.as_matrix() @ self.grp.rotation().matrix() @ (self.disp[3:] * self.K[3:])

        self.gg_ = (self.obj_0.inverse().compose(self.grp_0)).inverse().compose(self.obj.inverse().compose(self.grp))
        self.OU_noise += - self.OU_theta*self.OU_noise + self.OU_sigma*np.random.normal(size=6)