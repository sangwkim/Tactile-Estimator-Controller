#!/usr/bin/env python

try:
    import rospy
    from sensor_msgs.msg import CompressedImage
    from robot_comm.msg import *
    from robot_comm.srv import *
    from netft_rdt_driver.srv import Zero as ftZero
    from std_srvs.srv import *
    from std_msgs.msg import Float64MultiArray
    from geometry_msgs.msg import PoseStamped, WrenchStamped
    from visualization_msgs.msg import *
    is_ros = True
except ImportError:
    print('ros is not installed. Only replay and simulation can be done.')
    is_ros = False
    pass
import numpy as np
import time
from collections import deque
import sys
sys.path = sys.path[::-1]  # This is just a trick to enable Python2 (ROS) and Python3 (GTSAM) at the same time, if you are using newer ROS, you maybe don't have to do this.
import cv2, os, gtsam
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from networkmodels import DecoderFC, EncoderCNN, DecoderRNN
import gc
gc.collect()
from scipy.spatial.transform import Rotation as R

class tactile_module:

    def __init__(self, TCP_offset, r_convert, gtsam_st_prior=np.zeros(9), reach=62, verbose=False, isRNN=False, gt_collect=False):

        self.verbose = verbose # Prints measurement if True
        self.TCP_offset = TCP_offset
        self.r_convert = r_convert
        self.gtsam_st_prior = gtsam_st_prior # Prior of the grasp parameters. If zeros, use default value
        self.reach = reach # Prior on approximate object size
        self.gt_collect = gt_collect # Default is False. True when training the tactile module
        self.new_thres = 0.2 # threshold for logging new data
        self.cha_length = reach # characteristic length
        
        self.gtsam_on = False # True to turn on the factor graph
        self.new_added1 = False # True if new data is added
        self.openloop_mode = False # True if open loop mode
        self.gtsam_idle = True # True if the factor graph is done with update and ready to receive new data
        self.gtsam_updated = False # Flag to indicate that the factor graph is updated
        self.gtsam_i = 0 # Current imestep of the factor graph
        self.error_raise = False
        self.safety_flag = False
        self.safety_thres = 15 # Force (N) threshold to raise safety flag
        self.offset_calibrate = False # True to zero out the tatile displacement
        self.nn_output_offset = np.zeros(6) 
        self.cf = 0 # contact formation - 0: point / 1: line / 2: patch
        self.transition = False # contact formation transition

        self.cart_list, self.tact_list, self.com_rot_list = [], [], [] # buffer before transmitting to the factor graph (robot pose / tactile displacement / desired rotation)

        self.noenergyfactor = False # True: turns off the energy factor
        self.useonlinestiff = True # False: use fixed grasp parameters rather than online estimates
        self.min_type = 'force'
        self.min_type_dict = {'energy': 0., 'force_trn': 1., 'force': 2., 'distance': 3.}

        self.restart_gtsam = False
        self.restart1 = True
        self.restart2 = True
        self.isfresh1 = False
        self.isfresh2 = False
        self.isRNN = isRNN # True: use RNN model / False: use non-RNN model
        self.h_nc = None
        if not isRNN:
            self.load_nn_model()
        else:
            self.load_rnn_model()

        if is_ros:
            # Subscribers for the tactile images
            self.image_sub1 = rospy.Subscriber("/raspicam_node1/image/compressed",
                                            CompressedImage,
                                            self.call_back1,
                                            queue_size=1,
                                            buff_size=2**24)
            self.image_sub2 = rospy.Subscriber("/raspicam_node2/image/compressed",
                                            CompressedImage,
                                            self.call_back2,
                                            queue_size=1,
                                            buff_size=2**24)

            # Subscriber for the robot pose
            self.EGM_cart_sub = rospy.Subscriber("robot2_EGM/GetCartesian",
                                                PoseStamped, self.callback_cart)

            # Subscriber for the F/T sensor
            self.FT_sub = rospy.Subscriber("/netft/netft_data", WrenchStamped, self.callback_FT)

            self.everything_sub = rospy.Subscriber("/everything", Float64MultiArray, self.callback_gtsam, queue_size=1)

            self.gtsam_restart_pub = rospy.Publisher("/restart_cart", Float64MultiArray, queue_size=10)
            self.gtsam_add_new_pub = rospy.Publisher("/addnew_data", Float64MultiArray, queue_size=10)

        self.data1 = deque(maxlen=1000)
        self.data2 = deque(maxlen=1000)

        self.cart_init = np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float)
        self.cart_EGM = np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float)
        self.cart_EGM_ = np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float)

        self.m, self.n = 320, 427
        self.pad_m, self.pad_n = 160, 213

        self.g1_mean = 82.97
        self.g1_std = 47.74
        self.g2_mean = 76.44
        self.g2_std = 48.14

        self.nn_output = np.zeros(6)
        self.nn_output_ema = np.zeros(6)
        self.ema_decay_ = 0.9
        self.ema_decay = 0.99

        self.com_rot_queue = []

    def load_nn_model(self):

        model_dataset_name = 'new_sensor_011'
        self.xyzypr_limit = np.array([.25, .5, .5, .67/180*np.pi, .67/180*np.pi, 2./180*np.pi])

        save_name = "tactile_model"
        obj = 'all'
        obj = model_dataset_name + '_' + obj
        model_name = 'cnn_fc'
        save_model_path = "../weights/" + model_name + "/" + obj + '/'

        # EncoderCNN architecture
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
        in_channels = 3
        CNN_embed_dim = 512  # latent dim extracted by 2D CNN
        img_x, img_y = 300, 218  #
        dropout_cnn = 0.  # dropout probability

        # DecoderFC architecture
        FC_layer_nodes = [512, 512, 256]
        dropout_fc = 0.15
        k = 6  # output_dims (XYZYPR)

        use_cuda = torch.cuda.is_available()  # check if GPU exists
        self.device = torch.device(
            "cuda" if use_cuda else "cpu")  # use CPU or GPU

        # Create model
        self.cnn_encoder = EncoderCNN(img_x=img_x,
                                      img_y=img_y,
                                      input_channels=in_channels,
                                      fc_hidden1=CNN_fc_hidden1,
                                      fc_hidden2=CNN_fc_hidden2,
                                      drop_p=dropout_cnn,
                                      CNN_embed_dim=CNN_embed_dim).to(
                                          self.device)

        self.fc_decoder = DecoderFC(CNN_embed_dim=CNN_embed_dim,
                                    FC_layer_nodes=FC_layer_nodes,
                                    drop_p=dropout_fc,
                                    output_dim=k).to(self.device)

        self.cnn_encoder.load_state_dict(
            torch.load(save_model_path + save_name + '_cnn_encoder_best.pth'))
        self.fc_decoder.load_state_dict(
            torch.load(save_model_path + save_name + '_decoder_best.pth'))

        # Parallelize model to multiple GPUs
        if torch.cuda.device_count() > 0:
            print("Using", torch.cuda.device_count(), "GPUs!")
            if torch.cuda.device_count() > 1:
                self.cnn_encoder = nn.DataParallel(self.cnn_encoder)
                self.fc_decoder = nn.DataParallel(self.fc_decoder)

    def load_rnn_model(self):

        model_dataset_name = 'FG_000_210717_large'

        self.xyzypr_limit = np.array(
            [1., 2., 2., 2. / 180 * np.pi, 2. / 180 * np.pi, 4. / 180 * np.pi])

        save_name = "tactile_model"
        obj = 'all'
        obj = model_dataset_name + '_' + obj
        model_name = 'devel_cnn_lstm'
        save_model_path = "./weights/" + model_name + "/" + obj + '/'

        # EncoderCNN architecture
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
        in_channels = 3
        CNN_embed_dim = 512  # latent dim extracted by 2D CNN
        img_x, img_y = 300, 218  #
        dropout_cnn = 0.5  # dropout probability

        # DecoderRNN architecture
        RNN_hidden_layers = 2
        RNN_hidden_nodes = 512
        RNN_FC_dim = 256
        dropout_rnn = 0.5
        k = 6  # output_dims (XYZYPR)

        use_cuda = torch.cuda.is_available()  # check if GPU exists
        self.device = torch.device(
            "cuda" if use_cuda else "cpu")  # use CPU or GPU

        # Create model
        self.cnn_encoder = EncoderCNN(img_x=img_x,
                                      img_y=img_y,
                                      input_channels=in_channels,
                                      fc_hidden1=CNN_fc_hidden1,
                                      fc_hidden2=CNN_fc_hidden2,
                                      drop_p=dropout_cnn,
                                      CNN_embed_dim=CNN_embed_dim).to(
                                          self.device)

        self.rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim,
                                      h_RNN_layers=RNN_hidden_layers,
                                      h_RNN=RNN_hidden_nodes,
                                      h_FC_dim=RNN_FC_dim,
                                      drop_p=dropout_rnn,
                                      output_dim=k).to(self.device)

        self.cnn_encoder.load_state_dict(
            torch.load(save_model_path + save_name + '_cnn_encoder_best.pth'))
        self.rnn_decoder.load_state_dict(
            torch.load(save_model_path + save_name + '_decoder_best.pth'))

    def Matrix_to_Pose(self, m):
        p = gtsam.Pose3(gtsam.Rot3(m[:3,:3]), m[:3,-1])
        return p

    def Pose_to_Matrix(self, p):
        m = np.zeros((4,4))
        m[:3,-1] = p.translation()
        m[:3,:3] = p.rotation().matrix()
        return m

    # Receive and parse estimation and control plan computed by the factor graph
    def callback_gtsam(self, data):#waypoints, gtsam_i, ct_, ct_cov, obj_):
        d = np.asarray(data.data)
        self.gtsam_i = d[0]
        self.ct_ = self.Matrix_to_Pose(d[1:17].reshape(4,4))
        self.obj_ = self.Matrix_to_Pose(d[17:17+16].reshape(4,4))
        self.ct_cov = d[17+16:17+16+36].reshape(6,6)
        self.force_est = d[17+16+36:17+16+36+3]
        self.wr_ = d[17+16+36+3:17+16+36+3+6]
        self.disp_ = d[17+16+36+3+6:17+16+36+3+6+6]
        self.st = d[17+16+36+3+6+6:17+16+36+3+6+6+9]
        waypoints = d[17+16+36+3+6+6+9:-2].reshape(-1,16)
        if self.cf != d[-2]:
            self.cf = d[-2]
            self.transition = True
        else:
            self.transition = False
            self.waypoints_new = []
            for wp in waypoints:
                self.waypoints_new.append(self.Matrix_to_Pose(wp.reshape(4,4)))
            self.gtsam_idle = True
            self.gtsam_updated = True
        self.error_raise = d[-1]
        if self.gtsam_i > 0 and np.abs(self.ft[2]) < 0.05:
            self.error_raise = True

    # Receive and parse robot cartesian pose
    def callback_cart(self, data):
        p = data.pose.position
        o = data.pose.orientation
        cart_EGM_raw = np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w],
                                dtype=np.float)
        cart_EGM_raw[3:] = (R.from_quat(cart_EGM_raw[3:]) *
                            self.r_convert).as_quat()
        r = (R.from_quat(cart_EGM_raw[3:])).as_matrix()
        cart_EGM_raw[:3] += r.dot(-self.TCP_offset)
        self.cart_EGM = cart_EGM_raw.copy()
        self.current_pose = gtsam.Pose3(gtsam.Rot3(R.from_quat(self.cart_EGM[3:]).as_matrix()), self.cart_EGM[:3])
        if self.restart_gtsam:
            self.com_rot_queue = []
            zeroFTSensor = rospy.ServiceProxy('/netft/zero', ftZero)
            rospy.wait_for_service('/netft/zero', timeout=0.1)
            zeroFTSensor()
            self.data1.clear()
            self.data2.clear()
            self.cart_EGM_ = self.cart_EGM.copy()
            self.cf = 0
            print("restart gtsam")
            self.gtsam_idle = False
            #print(self.cart_EGM)
            self.gtsam_restart_pub.publish(Float64MultiArray(data=np.hstack((self.cart_EGM, self.gtsam_st_prior, self.reach, float(self.noenergyfactor), float(self.useonlinestiff), self.min_type_dict[self.min_type]))))
            #self.gtsam_graph.restart(self.cart_EGM)
            
            while True and self.openloop_mode == False:
                if self.gtsam_idle == True: break
            if self.openloop_mode == False:
                self.waypoints = [self.current_pose] + self.waypoints_new
                self.gtsam_updated = False
                self.s_list = [0]
                self.s, self.si = 0, 0
                for i in range(1, len(self.waypoints)):
                    lm = gtsam.Pose3.Logmap(self.waypoints[i-1].inverse().compose(self.waypoints[i]))
                    lmnorm = np.linalg.norm(np.hstack((lm[:3]*self.cha_length, lm[3:])))
                    self.s_list.append(self.s_list[-1] + lmnorm)
            self.cart_init = self.cart_EGM.copy()
            self.gtsam_on = True
            self.restart_gtsam = False
            print("restart complete")

    # callback for the force-torque sensor measurement
    def callback_FT(self, data):
        f = data.wrench.force
        t = data.wrench.torque
        self.ft = np.array([f.x, f.y, f.z, t.x, t.y, t.z], dtype=np.float)
        if np.abs(f.z) > self.safety_thres:
            self.safety_flag = True

    # callback for the Tactile Image from first sensor
    def call_back1(self, data):
        t = time.time()
        if is_ros:
            np_arr = np.fromstring(data.data, np.uint8)
            raw_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            raw_img = data

        img = raw_img[int(self.m / 2) - self.pad_m:int(self.m / 2) +
                      self.pad_m,
                      int(self.n / 2) - self.pad_n:int(self.n / 2) +
                      self.pad_n, :]
        img = cv2.resize(img, (300, 218)).astype(np.float32)

        img = (img - self.g1_mean) / self.g1_std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(np.expand_dims(img, 0), 0)

        if self.restart1:
            self.X1_0 = torch.from_numpy(img).type(torch.cuda.FloatTensor)
            self.nn_output, self.nn_output_ema = None, None
            self.h_nc = None
            self.safety_flag = False
            self.restart1 = False

        if self.offset_calibrate:
            self.nn_output_offset += self.nn_output_ema.copy()
            self.offset_calibrate = False

        self.X1 = torch.from_numpy(img).type(torch.cuda.FloatTensor)

        self.cnn_encoder.eval()
        if not self.isRNN:
            self.fc_decoder.eval()
        else:
            self.rnn_decoder.eval()

        self.X1_0 = self.X1_0.to(self.device)
        self.X1 = self.X1.to(self.device)

        self.isfresh1 = True

        if not self.gt_collect and self.isfresh1 and self.isfresh2 and not self.restart1 and not self.restart2:
            with torch.no_grad():
                if not self.isRNN:
                    nn_output = self.fc_decoder(
                        self.cnn_encoder(self.X1_0), self.cnn_encoder(self.X1),
                        self.cnn_encoder(self.X2_0), self.cnn_encoder(self.X2),
                        self.device).detach().cpu().numpy()[0, 0, :]
                else:
                    nn_output, self.h_nc = self.rnn_decoder.forward_single(
                        self.cnn_encoder(self.X1_0), self.cnn_encoder(self.X1),
                        self.cnn_encoder(self.X2_0), self.cnn_encoder(self.X2),
                        self.h_nc, self.device)
                    nn_output = nn_output.detach().cpu().numpy()[0, 0, :]
            nn_output *= 1.25 * self.xyzypr_limit
            nn_output[3:] *= 180 / np.pi
            nn_output -= self.nn_output_offset
            if self.nn_output is None:
                self.nn_output = nn_output.copy()
                self.nn_output_ema = nn_output.copy()
            else:
                self.nn_output *= self.ema_decay_
                self.nn_output += (1 - self.ema_decay_) * nn_output
                self.nn_output_ema *= self.ema_decay
                self.nn_output_ema += (1 - self.ema_decay) * nn_output
            self.isfresh1 = False
            self.isfresh2 = False

        if self.verbose:
            print(
                "{:+1.2f} {:+1.2f} {:+1.2f} {:+1.2f} {:+1.2f} {:+1.2f}".format(
                    self.nn_output[0], self.nn_output[1], self.nn_output[2],
                    self.nn_output[3], self.nn_output[4], self.nn_output[5]))

        # open loop mode (factor graph only used for estimation, factor graph's control part becomes redundant)
        if self.openloop_mode == True:
            d_rot = (R.from_quat(self.cart_EGM_[3:]).inv() *
                    R.from_quat(self.cart_EGM[3:])).as_rotvec()
            d_rot *= self.cha_length
            d_trn = self.cart_EGM[:3] - self.cart_EGM_[:3]
            delta = np.linalg.norm(np.hstack((d_rot, d_trn)))
            if delta > self.new_thres:
                self.cart_list.append(self.cart_EGM.copy())
                self.tact_list.append(self.nn_output.copy())
                self.com_rot_list.append(np.eye(3))
                self.cart_EGM_ = self.cart_EGM.copy()
                self.data1.append([raw_img, t, self.com_rot_list[-1], self.cart_list[-1], self.tact_list[-1], self.ft.copy(), None, None, False, self.force_est.copy(), self.wr_.copy(), self.disp_.copy(), self.Pose_to_Matrix(self.ct_).reshape(16), self.ct_cov.reshape(36)])
                self.new_added1 = True

        # if desired rotation sequence is not empty
        elif len(self.com_rot_queue) != 0 and len(self.s_list) > 1:
            if self.si < len(self.s_list)-1:
                if self.s > self.s_list[self.si+1]:
                    
                    self.si += 1
                    self.cart_list.append(self.cart_EGM.copy())
                    self.tact_list.append(self.nn_output.copy())
                    self.com_rot_list.append(self.com_rot_queue[0])
                    self.cart_EGM_ = self.cart_EGM.copy()

                    # add data to the buffer before sending to factor graph
                    self.data1.append([raw_img, t, self.com_rot_list[-1], self.cart_list[-1], self.tact_list[-1], self.ft.copy(),\
                                        [self.Pose_to_Matrix(p) for p in self.waypoints], self.si, False, self.force_est.copy(),\
                                            self.wr_.copy(), self.disp_.copy(), self.Pose_to_Matrix(self.ct_).reshape(16), self.ct_cov.reshape(36)])
                    self.com_rot_queue.pop(0)
                    self.new_added1 = True

        # if the update from the factor graph is received, update the control plan
        if not self.openloop_mode and self.gtsam_updated:
            self.gtsam_updated = False
            current_command = gtsam.Pose3(gtsam.Rot3(R.from_quat(self.cart_command[3:]).as_matrix()), self.cart_command[:3])

            self.waypoints = [current_command] + self.waypoints_new
            s_list_new = [0]
            self.s, self.si = 0, 0
            for i in range(1, len(self.waypoints)):
                lm = gtsam.Pose3.Logmap(self.waypoints[i-1].inverse().compose(self.waypoints[i]))
                lmnorm = np.linalg.norm(np.hstack((lm[:3]*self.cha_length, lm[3:])))
                s_list_new.append(s_list_new[-1] + lmnorm)
            self.s_list = s_list_new.copy()

        # if the factor graph is idle and buffer is not empty, transmit buffer to the factor graph
        if self.gtsam_idle and len(self.cart_list) != 0 and self.gtsam_on:
            if self.transition:
                for i in range(len(self.com_rot_list)):
                    self.data1[-1-i][2] = np.eye(3)
                self.com_rot_list = len(self.com_rot_list) * [np.eye(3)]
            self.gtsam_idle = False
            self.gtsam_add_new_pub.publish(Float64MultiArray(data=np.hstack((np.array(self.cart_list), np.array(self.tact_list), np.array(self.com_rot_list).reshape(-1,9))).flatten()))
            self.data1[-1][8] = True
            self.cart_list, self.tact_list, self.com_rot_list = [], [], []

    # callback for the Tactile Image from first sensor
    def call_back2(self, data):
        t = time.time()
        if is_ros:
            np_arr = np.fromstring(data.data, np.uint8)
            raw_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            raw_img = data

        img = raw_img[int(self.m / 2) - self.pad_m:int(self.m / 2) +
                      self.pad_m,
                      int(self.n / 2) - self.pad_n:int(self.n / 2) +
                      self.pad_n, :]
        img = cv2.resize(img, (300, 218)).astype(np.float32)

        img = (img - self.g2_mean) / self.g2_std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(np.expand_dims(img, 0), 0)

        if self.restart2:
            self.X2_0 = torch.from_numpy(img).type(torch.cuda.FloatTensor)
            self.restart2 = False

        self.X2 = torch.from_numpy(img).type(torch.cuda.FloatTensor)

        self.cnn_encoder.eval()
        if not self.isRNN:
            self.fc_decoder.eval()
        else:
            self.rnn_decoder.eval()

        self.X2_0 = self.X2_0.to(self.device)
        self.X2 = self.X2.to(self.device)

        self.isfresh2 = True

        if not self.gt_collect and self.isfresh1 and self.isfresh2 and not self.restart1 and not self.restart2:
            with torch.no_grad():
                if not self.isRNN:
                    nn_output = self.fc_decoder(
                        self.cnn_encoder(self.X1_0), self.cnn_encoder(self.X1),
                        self.cnn_encoder(self.X2_0), self.cnn_encoder(self.X2),
                        self.device).detach().cpu().numpy()[0, 0, :]
                else:
                    nn_output, self.h_nc = self.rnn_decoder.forward_single(
                        self.cnn_encoder(self.X1_0), self.cnn_encoder(self.X1),
                        self.cnn_encoder(self.X2_0), self.cnn_encoder(self.X2),
                        self.h_nc, self.device)
                    nn_output = nn_output.detach().cpu().numpy()[0, 0, :]
            nn_output *= 1.25 * self.xyzypr_limit
            nn_output[3:] *= 180 / np.pi
            nn_output -= self.nn_output_offset
            if self.nn_output is None:
                self.nn_output = nn_output.copy()
                self.nn_output_ema = nn_output.copy()
            else:
                self.nn_output *= self.ema_decay_
                self.nn_output += (1 - self.ema_decay_) * nn_output
                self.nn_output_ema *= self.ema_decay
                self.nn_output_ema += (1 - self.ema_decay) * nn_output
            self.isfresh1 = False
            self.isfresh2 = False

        if self.new_added1:
            self.data2.append(
                [raw_img, t, None, self.cart_EGM.copy(),
                 self.nn_output.copy(), self.ft.copy(), None, None, False, None, None, None, None, None])
            self.new_added1 = False

def main():
    print("start tactile module")
    rospy.init_node('tactile_module', anonymous=True)
    while not rospy.is_shutdown():
        tm = tactile_module(TCP_offset=np.array([-6.6, 0, -12.]),
                            r_convert=R.from_matrix([[0, 1, 0], [1, 0, 0],
                                                     [0, 0, -1]]),
                            verbose=True)
        rospy.spin()

if __name__ == "__main__":
    main()