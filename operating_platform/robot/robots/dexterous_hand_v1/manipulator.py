import zmq  
import threading  
import time  
import json
import numpy as np  
import torch  
import cv2  
from operating_platform.robot.robots.configs import DexterousHandRobotConfig 
from plot import DexterousHandVisualizer
  
ipc_address = "ipc:///tmp/dr-robot-dexterous-hand"  
  
context = zmq.Context()  
socket = context.socket(zmq.PAIR)  
socket.connect(ipc_address)  
socket.setsockopt(zmq.RCVTIMEO, 300)  
  
recv_wrist_data = {}  
recv_finger_data = {}  
recv_head_data = {}  
recv_images = {}  
lock = threading.Lock()  
  
# def recv_server():  
#     while True:  
#         try: 
       
#             message_parts = socket.recv_multipart()
#             # print(f"Received message with {len(message_parts)} parts", flush=True)   
#             if len(message_parts) < 2:  
#                 continue  
  
#             event_id = message_parts[0].decode('utf-8')  
            
#             buffer_bytes = message_parts[1]
#             #print(f"Event ID: {event_id}, Buffer size: {len(buffer_bytes)}")  
  
#             if 'wrist' in event_id:  
#                 pose_data = np.frombuffer(buffer_bytes, dtype=np.float32)
#                 # print(pose_data)  
#                 with lock:  
#                     recv_wrist_data[event_id] = pose_data  
  
#             elif 'finger' in event_id:  
#                 finger_data = np.frombuffer(buffer_bytes, dtype=np.float32)  
#                 with lock:  
#                     recv_finger_data[event_id] = finger_data  
  
#             elif 'head' in event_id:  
#                 head_data = np.frombuffer(buffer_bytes, dtype=np.float32)  
#                 with lock:  
#                     recv_head_data[event_id] = head_data  
  
#             elif 'image' in event_id:  
#                 img_array = np.frombuffer(buffer_bytes, dtype=np.uint8)   
#                 frame = (img_array.reshape((480, 640, 3)).copy())
#                 if frame is None:   
#                     with lock:    
#                         recv_images[event_id] = frame  
#                 else:  
#                     print(f"Failed to decode image for event_id: {event_id}, buffer size: {len(buffer_bytes)}") 
#                 # if frame is None:  
#                 #     # 如果 JPEG 解码失败，尝试作为原始 RGB 数据处理  
#                 #     if len(buffer_bytes) == 921600:  # 640*480*3  
#                 #         frame = img_array.reshape((480, 640, 3))  
#                 #         print(f"Processed as raw RGB data for event_id: {event_id}")  
#                 #     else:  
#                 #         print(f"Unknown image format for event_id: {event_id}, buffer size: {len(buffer_bytes)}")  
#                 #         continue  
      
#                 # if frame is not None:  
#                 #     # 确保是 BGR 格式，然后转换为 RGB  
#                 #     if frame.shape[-1] == 3:  
#                 #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
#                 #     with lock:  
#                 #         recv_images[event_id] = frame 
                 
  
#         except zmq.Again:  
#             continue  
#         except Exception as e:  
#             print("recv error:", e)  
#             break  
def recv_server():    
    while True:    
        try:   
            message_parts = socket.recv_multipart()  
            if len(message_parts) < 1:    
                continue    
    
            event_id = message_parts[0].decode('utf-8')  
            buffer_bytes = message_parts[1]  
            # print(len(buffer_bytes))
            # print(1)
            # 检查是否有第三个部分（metadata）  
            metadata = {}  
            if len(message_parts) >= 3:  
                try:  
                    metadata = json.loads(message_parts[2].decode('utf-8'))  
                except:  
                    metadata = {}  
    
            if 'wrist_left' in event_id:    
                pose_data = np.frombuffer(buffer_bytes, dtype=np.float32) 
                with lock:    
                    recv_wrist_data[event_id] = pose_data  
            elif 'wrist_right' in event_id:    
                pose_data = np.frombuffer(buffer_bytes, dtype=np.float32)  
                with lock:    
                    recv_wrist_data[event_id] = pose_data   
    
            elif 'finger_left' in event_id:    
                finger_data = np.frombuffer(buffer_bytes, dtype=np.float32)    
                with lock:    
                    recv_finger_data[event_id] = finger_data  
            elif 'finger_right' in event_id:    
                finger_data = np.frombuffer(buffer_bytes, dtype=np.float32)  
                with lock:    
                    recv_finger_data[event_id] = finger_data   
    
            elif 'head' in event_id:    
                head_data = np.frombuffer(buffer_bytes, dtype=np.float32)    
                with lock:    
                    recv_head_data[event_id] = head_data    
    
            elif 'image' in event_id:    
                img_array = np.frombuffer(buffer_bytes, dtype=np.uint8)  
                  
                # 从 metadata 获取图像信息，如果没有则使用默认值  
                encoding = metadata.get("encoding", "bgr8").lower()  
                width = metadata.get("width", 640)  
                height = metadata.get("height", 480)  
  
                if encoding == "bgr8":  
                    channels = 3  
                    frame = (  
                        img_array.reshape((height, width, channels))  
                        .copy()  # Copy So that we can add annotation on the image  
                    )  
                elif encoding == "rgb8":  
                    channels = 3  
                    frame = (img_array.reshape((height, width, channels)))  
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
                elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:  
                    channels = 3  
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  
                else:  
                    # 默认处理为 bgr8  
                    frame = (img_array.reshape((height, width, 3)).copy())  
                  
                # 最终转换为 RGB 格式存储  
                if frame is not None:  
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                    with lock:      
                        recv_images[event_id] = frame    
                else:    
                    print(f"Failed to decode image for event_id: {event_id}, buffer size: {len(buffer_bytes)}")  
                   
        except zmq.Again:    
            continue    
        except Exception as e:    
            print("recv error:", e)    
            break


class DexterousHandManipulator:  
    def __init__(self, config: DexterousHandRobotConfig):  
        self.config = config  
        self.robot_type = self.config.type  
        self.visualizer=DexterousHandVisualizer()
        #？
        self.cameras = {}  
        camera_index=1
        for key, features in self.camera_features.items():    
            camera_name = key.replace("observation.images.", "")  
  
            class MockCamera:  
                def __init__(self, shape, index):  
                    self.height = shape[0]  
                    self.width = shape[1]   
                    self.channels = shape[2]
                    self.camera_index = index
                    self.fps=30  
            
            self.cameras[camera_name] = MockCamera(features["shape"],camera_index)

        self.follower_arms = ["left", "right"] 
        self.logs = {}

        recv_thread = threading.Thread(target=recv_server, daemon=True)  
        recv_thread.start()   
        self.is_connected = False  
  
    @property  
    def camera_features(self):  
        return {  
            "observation.images.image": {  
                "shape": (480, 640, 3),  
                "names": ["height", "width", "channels"],  
                "info": None,  
            }  
        }  
  
    @property  
    def motor_features(self):  
        return {  
            "observation.state": {  
                "dtype": "float32",  
                "shape": (51,),  # 2手腕*7维 + 2手*15 + 1头部*7维 = 51维  
                "names": ["wrist_pose", "finger_joints", "head_pose"],  
            },  
            "action": {  
                "dtype": "float32",   
                "shape": (51,),  
                "names": ["wrist_pose", "finger_joints", "head_pose"],  
            }  
        }  
  
    @property  
    def features(self):  
        return {**self.motor_features, **self.camera_features}  
  
    def connect(self):  
        timeout = 10 
        start_time = time.perf_counter()  
  
        while True:  
            wrist_ready = all(key in recv_wrist_data for key in ['wrist_left', 'wrist_right'])  
            finger_ready = all(key in recv_finger_data for key in ['finger_left', 'finger_right'])  
            head_ready = 'head_pose' in recv_head_data  
            image_ready = 'image' in recv_images  
            print(wrist_ready,finger_ready,image_ready,head_ready)
            if wrist_ready and finger_ready and head_ready and image_ready:  
                break  
  
            if time.perf_counter() - start_time > timeout:  
                raise TimeoutError("设备连接超时")  
  
            time.sleep(0.1)  
  
        self.is_connected = True  
  
    
    def teleop_step(self, record_data=False):  
        if not self.is_connected:  
            raise Exception("设备未连接")  
    
        if not record_data:  
            return  
    
        with lock:  
            # 收集状态数据  
            state_data = []     
  
            if 'wrist_left' in recv_wrist_data:    
                state_data.extend(recv_wrist_data['wrist_left'][:7])  # 只取前7维  
            if 'wrist_right' in recv_wrist_data:    
                state_data.extend(recv_wrist_data['wrist_right'][:7])  # 只取前7维  
            if 'finger_left' in recv_finger_data:    
                state_data.extend(recv_finger_data['finger_left'][:15])  # 只取前15维  
            if 'finger_right' in recv_finger_data:    
                state_data.extend(recv_finger_data['finger_right'][:15])  # 只取前15维  
            if 'head_pose' in recv_head_data:    
                state_data.extend(recv_head_data['head_pose'][:7])  # 只取前7维 
                
            state = torch.from_numpy(np.array(state_data, dtype=np.float32))  

            # print(state)
            # 正确获取图像数据 - 遍历所有相机  
            images = {}  
            for name in self.cameras:  
                if name in recv_images:  
                    images[name] = torch.from_numpy(recv_images[name])  
                else:  
                    images[name] = torch.from_numpy(np.zeros((480, 640, 3), dtype=np.uint8))  
    
        obs_dict = {  
            "observation.state": state,  
        }  
        
        # 添加图像数据  
        for name in self.cameras:  
            obs_dict[f"observation.images.{name}"] = images[name]  
        
        action_dict = {  
            "action": state  
        }  
    
        return obs_dict, action_dict
  
    def disconnect(self):  
        self.is_connected = False
