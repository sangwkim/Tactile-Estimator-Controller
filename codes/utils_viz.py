import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation as R
import imageio
import sys
sys.path = sys.path[::-1]
import cv2
import gtsam
from stl import mesh
import copy

class visualization:
    
    def __init__(self, reach, pose_rel, height, T_og, view_elev=30, view_azim=45, view_center = (0,-25,-52+0), \
                    view_radius=30, object_name='rectangle', env_type='cone'):
        
        self.stage = 'headnote'
        self.pose_rel = pose_rel
        self.pose_trn = pose_rel[:3]
        self.pose_rot = R.from_euler('zyx', pose_rel[3:], degrees=False).as_matrix()
        self.T_wg = np.eye(4)
        self.T_wg[:3,:3] = self.pose_rot
        self.T_wg[:3,-1] = self.pose_trn
        self.T_og = T_og
        self.height = height
        self.view_elev = view_elev
        self.view_azim = view_azim
        self.view_center = view_center
        self.view_radius = view_radius
        self.object_name = object_name
        if 'obj' in self.object_name:
            self.model = mesh.Mesh.from_file(f'/home/devicereal/projects/cad_test/{self.object_name}.stl')
        elif 'magna' in self.object_name:
            self.model = mesh.Mesh.from_file(f'/home/devicereal/projects/magna_models/{self.object_name}.stl')
        self.env_type = env_type
        
        self.plotnum = 0
        self.fig = plt.figure(figsize=(8, 8), dpi=80)
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        
        if self.object_name in ['rectangle', 'circle', 'ellipse', 'circle_tight', 'ellipse_tight']:

            self.grp_x = [-17.5, -17.5, -17.5, -17.5, 17.5, 17.5, 17.5, 17.5]
            self.grp_y = [-12.5, 12.5, 12.5, -12.5, -12.5, 12.5, 12.5, -12.5]
            self.grp_z = [-12.5, -12.5, 12.5, 12.5, -12.5, -12.5, 12.5, 12.5]
            self.grp_v = [[0,1,2,3], [4,5,6,7]]
            self.grp_tupleList = list(zip(self.grp_x, self.grp_y, self.grp_z))
            
        elif self.object_name in ['hexagon', 'hexagon_tight']:

            self.grp_x = [-15.156, -15.156, -15.156, -15.156, 15.156, 15.156, 15.156, 15.156]
            self.grp_y = [-12.5, 12.5, 12.5, -12.5, -12.5, 12.5, 12.5, -12.5]
            self.grp_z = [-12.5, -12.5, 12.5, 12.5, -12.5, -12.5, 12.5, 12.5]
            self.grp_v = [[0,1,2,3], [4,5,6,7]]
            self.grp_tupleList = list(zip(self.grp_x, self.grp_y, self.grp_z))

        else:    

            self.grp_x = [-17.5, -17.5, -17.5, -17.5, 17.5, 17.5, 17.5, 17.5]
            self.grp_y = [-12.5, 12.5, 12.5, -12.5, -12.5, 12.5, 12.5, -12.5]
            self.grp_z = [-12.5, -12.5, 12.5, 12.5, -12.5, -12.5, 12.5, 12.5]
            self.grp_v = [[0,1,2,3], [4,5,6,7]]
            self.grp_tupleList = list(zip(self.grp_x, self.grp_y, self.grp_z))       
        
        if 'rectangle' in self.object_name:
            self.obj_x = [-17.5, 17.5, 17.5, -17.5, -17.5, 17.5, 17.5, -17.5]
            self.obj_z = [0, 0, 0, 0, 60, 60, 60, 60]
            self.obj_y = [-25, -25, 25, 25, -25, -25, 25, 25]
            self.obj_v = [[0,1,2,3], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7], [4,5,6,7]]
            self.obj_tupleList = list(zip(self.obj_x, self.obj_y, self.obj_z))
        elif 'hexagon' in self.object_name:
            self.obj_x = [15.156, 15.156, 0, -15.156, -15.156, 0, 15.156, 15.156, 0, -15.156, -15.156, 0]
            self.obj_z = [0, 0, 0, 0, 0, 0, 60, 60, 60, 60, 60, 60]
            self.obj_y = [-8.75, 8.75, 17.5, 8.75, -8.75, -17.5, -8.75, 8.75, 17.5, 8.75, -8.75, -17.5]
            self.obj_v = [[0,1,7,6], [1,2,8,7], [2,3,9,8], [3,4,10,9], [4,5,11,10], [5,0,6,11], [0,1,2,3,4,5], [6,7,8,9,10,11]]
            self.obj_tupleList = list(zip(self.obj_x, self.obj_y, self.obj_z))
        elif 'circle' in self.object_name:
            self.obj_x = 2*[17.5 * np.cos(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]
            self.obj_z = 40*[0]+40*[60]
            self.obj_y = 2*[17.5 * np.sin(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]
            self.obj_v = [[i,(i+1)%40,(i+1)%40+40,i+40] for i in range(40)] \
                           + [[i for i in range(40)]] + [[i+40 for i in range(40)]]
            self.obj_tupleList = list(zip(self.obj_x, self.obj_y, self.obj_z))
        elif 'ellipse' in self.object_name:
            self.obj_x = 2*[17.5 * np.cos(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]
            self.obj_z = 40*[0]+40*[60]
            self.obj_y = 2*[25 * np.sin(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]
            self.obj_v = [[i,(i+1)%40,(i+1)%40+40,i+40] for i in range(40)] \
                         + [[i for i in range(40)]] + [[i+40 for i in range(40)]]
            self.obj_tupleList = list(zip(self.obj_x, self.obj_y, self.obj_z))
        elif 'obj' in self.object_name or 'magna' in self.object_name:
            corners_dict = \
                {'rectangle': np.array([[-17.5,-25,0],[17.5,-25,0],[17.5,25,0],[-17.5,25,0]]),
                'hexagon': np.array([[15.156,-8.75,0],[15.156,8.75,0],[0,17.5,0],[-15.156,8.75,0],[-15.156,-8.75,0],[0,-17.5,0]]),
                    'obj_1': np.array([[0,22.134,0],[35,7.630,0],[35,36.329,0],[12.167,41.509,0]]) - np.array([17.5,27,0]),
                    'obj_2': np.empty((0,3)),
                    'obj_3': np.array([[3.919,8.398,0],[30.839,23.806,0],[3.919,42.91,0]]) - np.array([17.7,26,0]),
                    'obj_4': np.array([[0,15.615,0],[35,7.63,0],[35,43.870,0]]) - np.array([17.5,29,0]),
                    'obj_5': np.array([[0,25,4.055],[35,0,0],[35,50,1.986]]) - np.array([17.5,25,0]),
                    'obj_6': - np.array([[0,0,0],[35,14.079,0],[35,43.87,0],[0,32.352,0]]) + np.array([[17.5,26,0]]),
                    'obj_7': np.array([[0,0,0],[35,0,0],[35,50,0],[0,50,0]]) - np.array([17.5,25,0]),
                    'magna_1': np.empty((0,3)),
                    'magna_2': np.array([[1*np.cos(th), 1*np.sin(th),0] for th in np.arange(0,2*np.pi,0.1*np.pi)]),
                    'magna_3': np.array([[12*np.cos(th), 12*np.sin(th),0] for th in np.arange(0,2*np.pi,0.1*np.pi)]),
                    'magna_4': np.array([[-17.68,-14.33,0],[0,-32,0],[17.68,-14.33,0],[17.68,14.33,0],[0,32,0],[-17.68,14.33,0]]),
                    'magna_5': np.array([[1*np.cos(th), 1*np.sin(th),0] for th in np.arange(0,2*np.pi,0.1*np.pi)]),
                    'magna_6': np.array([[12*np.cos(th), 12*np.sin(th),0] for th in np.arange(0,2*np.pi,0.1*np.pi)]),
                    'magna_7': np.array([[-17.68,-14.33,0],[0,-32,0],[17.68,-14.33,0],[17.68,14.33,0],[0,32,0],[-17.68,14.33,0]]),
                    'magna_8': np.array([[1*np.cos(th), 1*np.sin(th),0] for th in np.arange(0,2*np.pi,0.1*np.pi)]),
                    'magna_9': np.array([[-5.4,-14.4,0],[5.4,-14.4,0],[5.4,14.4,0],[-5.4,14.4,0]])}
            corners = corners_dict[self.object_name]
            self.obj_x = list(corners[:,0])
            self.obj_y = list(corners[:,1])
            self.obj_z = list(corners[:,2])
            self.obj_v = [list(np.arange(len(corners)))]
            self.obj_tupleList = list(zip(self.obj_x, self.obj_y, self.obj_z))

        self.cpoint_x = [0]
        self.cpoint_y = [0]
        self.cpoint_z = [0]
        self.cpoint_tupleList = list(zip(self.cpoint_x, self.cpoint_y, self.cpoint_z))
        self.cline_x = [-50,50]
        self.cline_y = [0,0]
        self.cline_z = [0,0]
        self.cline_tupleList = list(zip(self.cline_x, self.cline_y, self.cline_z))
        
        if env_type == 'hole':
            if self.object_name in ['rectangle', 'ellipse']:
                self.env_x = [-19.75, 19.75, 19.75, -19.75, -19.75, 19.75, 19.75, -19.75, -100, 100, 100, -100]
                self.env_y = [-27.25, -27.25, 27.25, 27.25, -27.25, -27.25, 27.25, 27.25, -100, -100, 100, 100]
                self.env_z = [0, 0, 0, 0, -300, -300, -300, -300, 0, 0, 0, 0]
                self.env_v = [[0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7], [0,1,9,8], [1,2,10,9], [2,3,11,10], [3,0,8,11]]
            elif self.object_name == 'hexagon':
                self.env_x = [-17.406, 17.406, 17.406, -17.406, -17.406, 17.406, 17.406, -17.406, -100, 100, 100, -100]
                self.env_y = [-19.75, -19.75, 19.75, 19.75, -19.75, -19.75, 19.75, 19.75, -100, -100, 100, 100]
                self.env_z = [0, 0, 0, 0, -300, -300, -300, -300, 0, 0, 0, 0]
                self.env_v = [[0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7], [0,1,9,8], [1,2,10,9], [2,3,11,10], [3,0,8,11]]
            elif self.object_name == 'circle':
                self.env_x = [-19.75, 19.75, 19.75, -19.75, -19.75, 19.75, 19.75, -19.75, -100, 100, 100, -100]
                self.env_y = [-19.75, -19.75, 19.75, 19.75, -19.75, -19.75, 19.75, 19.75, -100, -100, 100, 100]
                self.env_z = [0, 0, 0, 0, -300, -300, -300, -300, 0, 0, 0, 0]
                self.env_v = [[0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7], [0,1,9,8], [1,2,10,9], [2,3,11,10], [3,0,8,11]]
            elif self.object_name == 'hexagon_tight':
                self.env_x = 2*[17.406, 17.406, 0, -17.406, -17.406, 0] + [5*x for x in [17.406, 17.406, 0, -17.406, -17.406, 0]]
                self.env_y = 2*[-10.049, 10.049, 20.098, 10.049, -10.049, -20.098] + [5*x for x in [-10.049, 10.049, 20.098, 10.049, -10.049, -20.098]]
                self.env_z = [0, 0, 0, 0, 0, 0, -300, -300, -300, -300, -300, -300, 0, 0, 0, 0, 0, 0]
                self.env_v = [[0,1,7,6], [1,2,8,7], [2,3,9,8], [3,4,10,9], [4,5,11,10], [5,0,6,11], [0,1,13,12], [1,2,14,13], [2,3,15,14], [3,4,16,15], [4,5,17,16], [5,0,12,17]]
            elif self.object_name == 'circle_tight':
                self.env_x = 2*[19.75 * np.cos(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)] + [5*x for x in [19.75 * np.cos(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]]
                self.env_y = 2*[19.75 * np.sin(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)] + [5*x for x in [19.75 * np.sin(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]]
                self.env_z = 40*[0]+40*[-300]+40*[0]
                self.env_v = [[i,(i+1)%40,(i+1)%40+40,i+40] for i in range(40)] \
                             + [[i,(i+1)%40,(i+1)%40+80,i+80] for i in range(40)] \
                                 + [[i for i in range(40)]]
            elif self.object_name == 'ellipse_tight':
                self.env_x = 2*[19.75 * np.cos(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)] + [5*x for x in [19.75 * np.cos(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]]
                self.env_y = 2*[27.25 * np.sin(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)] + [5*x for x in [19.75 * np.sin(th) for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)]]
                self.env_z = 40*[0]+40*[-300]+40*[0]
                self.env_v = [[i,(i+1)%40,(i+1)%40+40,i+40] for i in range(40)] \
                             + [[i,(i+1)%40,(i+1)%40+80,i+80] for i in range(40)] \
                                 + [[i for i in range(40)]]
            self.env_tupleList = list(zip(self.env_x, self.env_y, self.env_z))
        elif env_type == 'wall':
            self.env_x = [-100, -100, -100, 100, 100, 100]
            self.env_y = [1, 1, 100, 1, 1, 100]#[-25, -25, -125, -25, -25, -125]
            self.env_z = [-100, 0, 0, -100, 0, 0]
            self.env_v = [[0,1,4,3], [1,2,5,4]]
            self.env_tupleList = list(zip(self.env_x, self.env_y, self.env_z))
        elif env_type == 'pedestal':
            self.env_x = [0, 400, 0, 0]
            self.env_y = [0, 0, -400, 0]
            self.env_z = [0, 0, 0, -400]
            self.env_v = [[0,1,2],[0,2,3],[0,3,1]]
            #self.env_x = [10, 0, -10, 10, 0, -10]
            #self.env_y = [20, -20, 20, 20, -20, 20]#[-25, -25, -125, -25, -25, -125]
            #self.env_z = [0, -8.11, -20, -100, -100, -100]
            #self.env_v = [[0,1,4,3],[1,2,5,4],[2,0,3,5]]#[[0,1,2],[0,1,4,3],[1,2,5,4],[2,0,3,5]]
            self.env_tupleList = list(zip(self.env_x, self.env_y, self.env_z))
        elif env_type == 'cone':
            self.env_x = [0, 400, 0, 0]
            self.env_y = [0, 0, -400, 0]
            self.env_z = [0, 0, 0, -400]
            self.env_v = [[0,1,2],[0,2,3],[0,3,1]]
            """
            self.env_x = [0, -30, 30, 30, -30]
            self.env_y = [0, -30, -30, 30, 30]
            self.env_z = [0, -60, -60, -60, -60]
            self.env_v = [[0,1,2],[0,2,3],[0,3,4],[0,4,1]]
            """
            self.env_tupleList = list(zip(self.env_x, self.env_y, self.env_z))
        elif env_type == 'floor':
            self.env_x = [-400, 400, 400, -400]
            self.env_y = [-400, -400, 400, 400]
            self.env_z = [0, 0, 0, 0]
            self.env_v = [[0,1,2,3]]
            self.env_tupleList = list(zip(self.env_x, self.env_y, self.env_z))

        if env_type != 'floor':
            self.plot_environment((0,0,-self.height), np.eye(3), alpha=0.1)
            T_og_ = self.T_og.copy()
            T_og_[:3,-1] = np.array([0,0,-reach])
            T_wo_ = self.T_wg @ np.linalg.inv(T_og_)
            self.plot_object(T_wo_[:3,-1], T_wo_[:3,:3], alpha=0.1)
        self.plot_gripper(self.pose_trn, self.pose_rot, alpha=0.1)
        T_wo = self.T_wg @ np.linalg.inv(self.T_og)
        self.plot_object(T_wo[:3,-1], T_wo[:3,:3], alpha=0.1)
        self.plot_cpoint(self.pose_trn + self.pose_rot.dot(np.array([0,0,-reach])), self.pose_rot)
        self.plot_cline(self.pose_trn + self.pose_rot.dot(np.array([0,0,-reach])), self.pose_rot)   

        self.show_env = True
        self.show_grp = True
        self.show_obj_gt = True
        self.show_obj_est = True
        self.show_cline = True
        
        self.images, self.images_command = [], []
        
        self.plot_show()
        
    def set_show(self, env=True, grp = True, obj_gt=True, obj_est=True, cpoint=True, cline=False):
        
        self.show_env = env
        self.show_grp = grp
        self.show_obj_gt = obj_gt
        self.show_obj_est = obj_est
        self.show_cpoint = cpoint
        self.show_cline = cline
        
    def set_viewpoint(self, elev, azim):#elev=0, azim=0): #elev=30, azim=-45):
        
        self.ax.view_init(elev=elev, azim=azim)
        
    def plot_clear(self):
        
        self.ax.clear()
    
    def plot_update(self, grp_trn, grp_rot, obj_trn, obj_rot, c_trn, c_rot, obj_gt_trn, obj_gt_rot, alpha=0.25):
        
        self.plotnum += 1
        
        if self.show_env:
            self.plot_environment((0,0,-self.height), np.eye(3)) 
        if self.show_cpoint:
            self.plot_cpoint(c_trn, c_rot) 
        if self.show_cline:
            self.plot_cline(c_trn, c_rot) 
        if self.show_grp:
            self.plot_gripper(grp_trn, grp_rot, alpha=alpha)
        if self.show_obj_est:
            self.plot_object(obj_trn, obj_rot, alpha=alpha)
        if self.show_obj_gt:
            self.plot_object(obj_gt_trn, obj_gt_rot, alpha=alpha)
        
        self.plot_show()
        
    def plot_gripper(self, trn, rot, s=0, lw=1, alpha=0.25, fc='b', ec='b'):

        trn = self.pose_trn + self.pose_rot.dot(trn)
        rot = self.pose_rot.dot(rot)
        
        tupleList = [rot.dot(point) + trn for point in self.grp_tupleList]
        #tupleList = [self.pose_rot.dot(point) + self.pose_trn for point in tupleList]
        
        poly3d = self.form_poly(tupleList, self.grp_v)
        self.ax.scatter(np.array(tupleList)[:,0], np.array(tupleList)[:,1], np.array(tupleList)[:,2], s=s)
        self.ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fc, ec=ec, linewidths=lw, alpha=alpha))
        
    def plot_object(self, trn, rot, s=0, lw=1, alpha=0.25, fc='b', ec='b'):

        trn = self.pose_trn + self.pose_rot.dot(trn)
        rot = self.pose_rot.dot(rot)
        
        tupleList = [rot.dot(point) + trn for point in self.obj_tupleList]
        #tupleList = [self.pose_rot.dot(point) + self.pose_trn for point in tupleList]
        
        if self.object_name == 'rectangle':        
            poly3d = self.form_poly(tupleList, self.obj_v)
        elif self.object_name == 'hexagon':
            poly3d = self.form_poly(tupleList, self.obj_v[:6])
            poly3d_ = self.form_poly(tupleList, self.obj_v[6:])
            self.ax.add_collection3d(Poly3DCollection(poly3d_, facecolors=fc, ec=ec, linewidths=0, alpha=alpha))
        elif self.object_name == 'circle':
            poly3d = self.form_poly(tupleList, self.obj_v[-2:])
            poly3d_ = self.form_poly(tupleList, self.obj_v[:-2])
            self.ax.add_collection3d(Poly3DCollection(poly3d_, facecolors=fc, ec=ec, linewidths=0.1, alpha=alpha))
        elif self.object_name == 'ellipse':
            poly3d = self.form_poly(tupleList, self.obj_v[-2:])
            poly3d_ = self.form_poly(tupleList, self.obj_v[:-2])
            self.ax.add_collection3d(Poly3DCollection(poly3d_, facecolors=fc, ec=ec, linewidths=0.1, alpha=alpha))
        elif 'obj' in self.object_name or 'magna' in self.object_name:
            poly3d = self.form_poly(tupleList, self.obj_v)
            model = copy.deepcopy(self.model)
            model.rotate_using_matrix(np.linalg.inv(rot))
            model.translate(trn)
            print(model)
            self.ax.add_collection3d(Poly3DCollection(model.vectors, facecolors=fc, ec=ec, linewidths=0, alpha=alpha))

        if len(tupleList) != 0:
            self.ax.scatter(np.array(tupleList)[:,0], np.array(tupleList)[:,1], np.array(tupleList)[:,2], s=s)
            self.ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fc, ec=ec, linewidths=lw, alpha=alpha))
        
    def plot_environment(self, trn, rot, s=1, lw=1, alpha=0.25,ec='r',fc='y'):
        
        tupleList = [rot.dot(point) + trn for point in self.env_tupleList]
        
        if self.env_type == 'hole':
            if self.object_name == 'rectangle':
                poly3d = self.form_poly(tupleList, self.env_v[:4])
                poly3d_ = self.form_poly(tupleList, self.env_v[4:])
            elif self.object_name == 'hexagon_tight':
                poly3d = self.form_poly(tupleList, self.env_v[:6])
                poly3d_ = self.form_poly(tupleList, self.env_v[6:])
            elif self.object_name == 'circle_tight':
                poly3d = self.form_poly(tupleList, self.env_v[-1:])
                poly3d_ = self.form_poly(tupleList, self.env_v[:-1])
            elif self.object_name == 'ellipse_tight':
                poly3d = self.form_poly(tupleList, self.env_v[-1:])
                poly3d_ = self.form_poly(tupleList, self.env_v[:-1])
            self.ax.scatter(np.array(tupleList)[:,0], np.array(tupleList)[:,1], np.array(tupleList)[:,2], s=s)
            self.ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fc, ec=ec, linewidths=lw, alpha=0.12))
            self.ax.add_collection3d(Poly3DCollection(poly3d_, facecolors=fc, ec=ec, linewidths=0, alpha=alpha))
        else:#elif self.env_type == 'wall':
            poly3d = self.form_poly(tupleList, self.env_v)
            self.ax.scatter(np.array(tupleList)[:,0], np.array(tupleList)[:,1], np.array(tupleList)[:,2], s=s)
            self.ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fc, ec=ec, linewidths=lw, alpha=alpha))
        
    def plot_cline(self, trn, rot, linestyle='-', lw=4, c='r', alpha=1):

        trn = self.pose_trn + self.pose_rot.dot(trn)
        rot = self.pose_rot.dot(rot)
        
        tupleList = [rot.dot(point) + trn for point in self.cline_tupleList]
        #tupleList = [self.pose_rot.dot(point) + self.pose_trn for point in tupleList]
        points = np.array(tupleList)
        
        self.ax.plot3D(points[:,0], points[:,1], points[:,2], linestyle, lw=lw, c=c, alpha=alpha)

    def plot_cpoint(self, trn, rot, linestyle='o-', lw=4, c='r', alpha=1, markersize=20):

        trn = self.pose_trn + self.pose_rot.dot(trn)
        rot = self.pose_rot.dot(rot)
        
        tupleList = [rot.dot(point) + trn for point in self.cpoint_tupleList]
        #tupleList = [self.pose_rot.dot(point) + self.pose_trn for point in tupleList]
        points = np.array(tupleList)
        
        self.ax.plot3D(points[:,0], points[:,1], points[:,2], linestyle, lw=lw, c=c, alpha=alpha, markersize=markersize)

    def plot_torque(self, origin, vec, scale, scale_=2.5, lw=4, c='k'):
        vec *= -1
        mag = scale * np.linalg.norm(vec)
        trn = self.pose_trn + self.pose_rot.dot(origin)
        traj = []
        traj.append([mag,0,0])
        traj.append([mag+0.6*0.3*mag,0.3*mag,0])
        traj.append([mag,0,0])
        traj.append([mag-0.6*0.3*mag,0.3*mag,0])
        traj = np.array(traj + [[mag*np.cos(i), mag*np.sin(i), 0] for i in np.arange(0,1.75*np.pi,0.02*np.pi)])
        rot = np.eye(3)
        rot[:,2] = vec / np.linalg.norm(vec)
        rot[:,0] = np.cross(np.array([0,0,1]), vec)
        if np.linalg.norm(rot[:,0]) != 0:
            rot[:,0] /= np.linalg.norm(rot[:,0])
        else:
            rot[:,0] = np.array([1,0,0])
        rot[:,1] = np.cross(rot[:,2], rot[:,0])
        traj = (rot @ traj.T).T + trn
        self.ax.plot3D(traj[:,0], traj[:,1], traj[:,2], lw=lw, c=c)
        self.ax.quiver(trn[0]+scale_*vec[0], trn[1]+scale_*vec[1], trn[2]+scale_*vec[2], -scale_*2*vec[0], -scale_*2*vec[1], -scale_*2*vec[2], alpha=0.3, color=c, lw=10, arrow_length_ratio=0.3)
        
    def plot_confidence_ellipsoid(self, bias, cov, scale=1, rstride=4, cstride=4, color='r', alpha=0.2):
        
        bias = self.pose_trn + self.pose_rot.dot(bias)
        cov = self.pose_rot @ cov @ self.pose_rot.T

        bias = np.expand_dims(bias,1)
        cov *= scale**2

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        ellipsoid = (cov @ np.stack((x, y, z), 0).reshape(3, -1) + bias).reshape(3, *x.shape)

        self.ax.plot_surface(*ellipsoid, rstride=rstride, cstride=cstride, color=color, alpha=alpha)

    def plot_confidence_cone(self, trn, rot, cov, scale=1, color='r', alpha=0.2):

        trn = self.pose_trn + self.pose_rot.dot(trn)
        rot = self.pose_rot @ rot
        vx, vy, vz = rot[:,0], rot[:,1], rot[:,2]
        sy, sz = np.clip(scale*cov[1,1]**.5,0,1.569), np.clip(scale*cov[2,2]**.5,0,1.569)
        l = np.linspace(-40,40,80)
        theta = np.linspace(0,2*np.pi,80)
        l, theta = np.meshgrid(l, theta)
        r = np.linspace(-40,40,80)
        X, Y, Z = [trn[i] + vx[i] * l + r * np.tan(sz)
               * np.sin(theta) * vy[i] + r * np.tan(sy) * np.cos(theta) * vz[i] for i in [0, 1, 2]]
        self.ax.plot_surface(X, Y, Z, rstride=4, cstride=4, color=color, linewidth=0, antialiased=False, alpha=alpha)
        
    def plot_coordinate_axis(self, trn, rot, linestyle='-', lw=4, c='r', alpha=1, scale=20, ar=.0):
        
        trn = self.pose_rot.dot(trn) + self.pose_trn
        rot = self.pose_rot @ rot
        
        self.ax.quiver(trn[0], trn[1], trn[2], scale*rot[0,0], scale*rot[1,0], scale*rot[2,0], color=c, lw=lw, alpha=alpha, arrow_length_ratio=ar)
        self.ax.quiver(trn[0], trn[1], trn[2], scale*rot[0,1], scale*rot[1,1], scale*rot[2,1], color=c, lw=lw, alpha=alpha, arrow_length_ratio=ar)
        self.ax.quiver(trn[0], trn[1], trn[2], scale*rot[0,2], scale*rot[1,2], scale*rot[2,2], color=c, lw=lw, alpha=alpha, arrow_length_ratio=ar)

    def plot_coordinate_axis_2(self, pose, linestyle='-', lw=2, c='r', alpha=1, scale=25):

        trn = self.pose_rot.dot(pose.translation()) + self.pose_trn
        rot = self.pose_rot @ pose.rotation().matrix()
        
        #self.ax.quiver(trn[0], trn[1], trn[2], scale*rot[0,0], scale*rot[1,0], scale*rot[2,0], color=c, lw=lw)#, arrow_length_ratio=ar)
        #self.ax.quiver(trn[0], trn[1], trn[2], scale*rot[0,1], scale*rot[1,1], scale*rot[2,1], color=c, lw=lw)
        self.ax.quiver(trn[0], trn[1], trn[2], scale*rot[0,2], scale*rot[1,2], scale*rot[2,2], color=c, lw=lw, arrow_length_ratio=0.)
    
    def form_poly(self, tupleList, vertices):
        
        poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
        return poly3d    
    
    def set_axes_equal(self, plot_center=(0,-25,-52+0), plot_radius=30):

        x_middle = plot_center[0]
        y_middle = plot_center[1]
        z_middle = plot_center[2]

        self.ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.ax.set_zlim3d([z_middle - 0.85*plot_radius, z_middle + 0.85*plot_radius])
    
    def plot_show(self):
        
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.axis('off')
        self.set_axes_equal(self.view_center, self.view_radius)
        self.set_viewpoint(self.view_elev, self.view_azim)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.savefig(f'../../plots/plot_{self.plotnum}.png')
        #image = cv2.imread(f'../../plots/plot_{self.plotnum}.png')  
        #image =cv2.putText(img=np.copy(image), text=self.stage, org=(260,70),fontFace=1, fontScale=1.5, color=(0,200,0), thickness=2)
        #cv2.imwrite(f'../../plots/plot_{self.plotnum}.png', image)
        self.images.append(imageio.imread(f'../../plots/plot_{self.plotnum}.png'))

    def plot_show_command(self):
        
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.axis('off')
        self.set_axes_equal(np.zeros(3), 15)
        self.set_viewpoint(self.view_elev, self.view_azim)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.savefig(f'../../plots/command.png')
        image = cv2.imread(f'../../plots/command.png')
        #image = cv2.putText(img=np.copy(image), text='command rotation', org=(400,750),fontFace=3, fontScale=1.5, color=tuple(int('1f77b4'[i:i+2], 16) for i in (4,2,0)), thickness=2)
        #image = cv2.putText(img=np.copy(image), text='actual rotation', org=(400,700),fontFace=3, fontScale=1.5, color=tuple(int('2ca02c'[i:i+2], 16) for i in (4,2,0)), thickness=2)
        cv2.imwrite(f'../../plots/command.png', image)
        self.images_command.append(imageio.imread(f'../../plots/command.png'))