import matplotlib.pyplot as plt  
import numpy as np  
from mpl_toolkits.mplot3d import Axes3D  
from scipy.spatial.transform import Rotation as R  
  
class DexterousHandVisualizer:  
    def __init__(self):  
        self.fig = plt.figure(figsize=(12, 8))  
        self.ax = self.fig.add_subplot(111, projection='3d')  
        self.setup_plot()  
          
    def setup_plot(self):  
        self.ax.set_xlabel('X')  
        self.ax.set_ylabel('Y')   
        self.ax.set_zlabel('Z')  
        self.ax.set_title("Dexterous Hand Pose Visualization")  
          
    def update_poses(self, recv_wrist_data, recv_head_data):  
        self.ax.cla()  
        self.setup_plot()  
          
        # 可视化左右手腕位置  
        if 'wrist_left' in recv_wrist_data:  
            pos = recv_wrist_data['wrist_left'][:3]  # 位置  
            rot = recv_wrist_data['wrist_left'][3:7]  # 四元数  
            self.plot_pose(pos, rot, 'cyan', 'Left Wrist')  
              
        if 'wrist_right' in recv_wrist_data:  
            pos = recv_wrist_data['wrist_right'][:3]  
            rot = recv_wrist_data['wrist_right'][3:7]  
            self.plot_pose(pos, rot, 'magenta', 'Right Wrist')  
              
        # 可视化头部姿态  
        if 'head_pose' in recv_head_data:  
            pos = recv_head_data['head_pose'][:3]  
            rot = recv_head_data['head_pose'][3:7]  
            self.plot_pose(pos, rot, 'green', 'Head')  
              
        plt.draw()  
        plt.pause(0.01)  
          
    def plot_pose(self, position, quaternion, color, label):  
        # 绘制坐标系  
        origin = position  
        rot_matrix = R.from_quat(quaternion).as_matrix()  
          
        axis_length = 0.1  
        x_axis = origin + rot_matrix[:, 0] * axis_length  
        y_axis = origin + rot_matrix[:, 1] * axis_length    
        z_axis = origin + rot_matrix[:, 2] * axis_length  
          
        self.ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], z_axis[2]],   
                    c=color, linewidth=2, alpha=0.8)  
        self.ax.text(*origin, label, color=color, fontsize=8)