from gevent import monkey
monkey.patch_all()

from flask import Flask, jsonify, Response, request, send_file, session, redirect, url_for, render_template
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
import io
import os
import logging
import datetime
from flask_socketio import SocketIO, emit
import json


class VideoStream:
    def __init__(self, stream_id, stream_name):
        self.stream_id = stream_id
        self.name = stream_name
        self.running = False
        self.frame_buffers = [None, None]  # 双缓冲
        self.buffer_index = 0
        self.lock = threading.Lock()
        

    def start(self):
        """启动视频流（仅标记为运行）"""
        if self.running:
            print(f"已经启动视频流")
            return True
        self.running = True
        return True
    
    def stop(self):
        """停止视频流"""
        self.running = False

    def update_frame(self, frame_data):
        """接收外部帧数据并更新当前帧"""
        if not self.running:
            return
        
        # 解码图像
        img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        
        # 压缩图像（可选）
        img = cv2.resize(img, (640, 480))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, jpeg = cv2.imencode('.jpg', img, encode_param)
        compressed_frame = jpeg.tobytes()

        with self.lock:
            self.buffer_index = 1 - self.buffer_index
            self.frame_buffers[self.buffer_index] = compressed_frame

    def get_frame(self):
        if not self.running:
            return self.generate_blank_frame()
        with self.lock:
            return self.frame_buffers[self.buffer_index]
    
    @staticmethod 
    def generate_blank_frame():
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode('.jpg', blank)
        return jpeg.tobytes()


class FlaskServer:
    def __init__(self):
        # 初始化Flask应用
        self.app = Flask(__name__)
        self.app.secret_key = 'agilex'  # 暂时密钥
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        CORS(self.app)
        
        # 初始化日志
        self.init_logging()
        
        # 初始化实例变量
        self.robot_sid = None
        self.video_list = {}
        self.video_timestamp = time.time()
        self.video_streams = {}
        self.stream_status = {}
        self.frame_lock = threading.Lock()
        self.init_streams_flag = False
        self.task_steps = {}
        
        # 响应模板
        self.response_start_collection = {
            "timestamp": time.time(),
            "msg": None
        }
        self.response_finish_collection = {
            "timestamp": time.time(),
            "msg": None,
            "data": None
        }
        self.response_submit_collection = {
            "timestamp": time.time(),
            "msg": None
        }
        self.response_discard_collection = {
            "timestamp": time.time(),
            "msg": None
        }
        
        # 注册路由
        self.register_routes()
    
    def init_logging(self):
        """初始化日志配置"""
        now = datetime.datetime.now()
        file_name = "./log/" + now.strftime("%Y.%m.%d.%H.%M") + ".log"
        log_dir = "./log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        logging.basicConfig(
            filename=file_name,
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode="a"
        )
    
    def register_routes(self):
        """注册所有路由"""
        # 系统信息
        self.app.add_url_rule('/api/info', 'system_info', self.system_info, methods=['GET'])
        
        # 视频流管理
        self.app.add_url_rule('/api/stream_info', 'get_streams', self.get_streams, methods=['GET'])
        self.app.add_url_rule('/api/start_stream', 'start_stream', self.start_stream, methods=['POST'])
        self.app.add_url_rule('/api/get_stream/<stream_id>', 'stream_video', self.stream_video, methods=['GET'])
        self.app.add_url_rule('/api/stop_stream/<stream_id>', 'stop_stream', self.stop_stream, methods=['POST'])
        
        # 采集控制
        self.app.add_url_rule('/api/start_collection', 'start_collection', self.start_collection, methods=['POST'])
        self.app.add_url_rule('/api/finish_collection', 'finish_collection', self.finish_collection, methods=['POST'])
        self.app.add_url_rule('/api/discard_collection', 'discard_collection', self.discard_collection, methods=['POST'])
        self.app.add_url_rule('/api/submit_collection', 'submit_collection', self.submit_collection, methods=['POST'])
       
        
        # 机器人接口
        self.app.add_url_rule('/robot/update_stream/<stream_id>', 'update_frame', self.update_frame, methods=['POST'])
        self.app.add_url_rule('/robot/stream_info', 'robot_get_video_list', self.robot_get_video_list, methods=['POST'])
        self.app.add_url_rule('/robot/response', 'robot_response', self.robot_response, methods=['POST'])
        self.app.add_url_rule('/robot/get_task_steps', 'get_task_steps', self.get_task_steps, methods=['GET'])
        
        # WebSocket事件
        self.socketio.on_event('connect', self.handle_connect)
        self.socketio.on_event('HEARTBEAT', self.handle_heartbeat)
        self.socketio.on_event('disconnect', self.handle_disconnect)
    
    def send_message_to_robot(self, sid, message):
        """向特定机器人客户端发送消息"""
        self.socketio.emit('robot_command', message, room=sid, namespace='/')

    
    # ---------------------- WebSocket 事件处理 ----------------------
    def handle_connect(self):
        self.robot_sid = request.sid
        print('Client connected')
    
    def handle_heartbeat(self):
        """响应心跳包"""
        emit('HEARTBEAT_RESPONSE', {'server': 'alive'})
    
    def handle_disconnect(self):
        print('Client disconnected')
    
    # ---------------------- 路由处理方法 ----------------------
    def system_info(self):
        """获取系统信息"""
        active_count = sum(1 for s in self.stream_status.values() if s["active"])
        
        return jsonify({
            "status": "running",
            "streams_active": active_count,
            "total_streams": len(self.stream_status),
            "timestamp": time.time(),
            "streams": self.stream_status
        })
    
    def get_streams(self):
        try:
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'video_list'})
            """获取可用视频流列表"""
            now_time = time.time()
            while True:
                if 0 < self.video_timestamp - now_time < 2:
                    return jsonify(self.video_list), 200
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 5: # 正式环境设为2.5超时
                    return jsonify({"msg": "机器人响应超时"}), 404          
        except Exception as e:
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def start_stream(self):
        try:
            """启动指定视频流"""
            data = request.get_json()
            stream_id = data.get('stream_id')
            if stream_id not in self.video_streams:
                return jsonify({"error": "无效的视频流ID"}), 404
                
            success = self.video_streams[stream_id].start()
            if success:
                self.stream_status[stream_id]["active"] = True
                return jsonify({"msg": "success"}), 200
            else:
                return jsonify({"error": "启动视频流失败"}), 500
        except Exception as e:
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def stream_video(self, stream_id):
        try:
            # 尝试将 stream_id 转换为整数
            stream_id = int(stream_id)
        except ValueError:
            return jsonify({"error": "无效的流ID,必须为数字"}), 400
        
        if stream_id not in self.video_streams:
            return jsonify({"error": "视频流不存在"}), 404
      
        if not self.video_streams[stream_id].running:
            return jsonify({"error":"视频流未开启"}), 404
        
        def generate():
            max_retries = 10  # 最大重试次数
            retry_count = 0
            try:
                while True:
                    frame = self.video_streams[stream_id].get_frame()
                    if frame:
                        retry_count = 0  # 重置重试计数器
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    else:
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"[WARNING] 视频流 {stream_id} 无法获取帧，超过最大重试次数")
                            break  # 退出循环
                    time.sleep(0.03)  # 控制帧率
            except GeneratorExit:
                print(f"[INFO] 客户端断开视频流: {stream_id}")
    
        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    
    def stop_stream(self, stream_id):
        try:
            # 尝试将 stream_id 转换为整数
            stream_id = int(stream_id)
        except ValueError:
            return jsonify({"error": "无效的流ID,必须为数字"}), 400
        
        """停止指定视频流"""
        if stream_id not in self.video_streams:
            return jsonify({"error": "视频流不存在"}), 404
            
        self.video_streams[stream_id].stop()
        self.stream_status[stream_id]["active"] = False
        
        return jsonify({"status": "stopped"})
    
    def init_streams(self):
        """初始化视频流"""
        if 'streams' in self.video_list:
            for stream in self.video_list['streams']:
                self.video_streams[stream['id']] = VideoStream(stream['id'], stream['name'])
                self.stream_status[stream['id']] = {
                    "name": str(stream['name']),
                    "active": False
                }
        print(self.video_list)
    
    def start_collection(self):
        try:
            data = request.get_json()
            self.task_steps = data
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'start_collection','msg': data})
            while True:
                if 0 < self.response_start_collection["timestamp"] - now_time < 2:
                    if self.response_start_collection['msg'] == "success":
                        return jsonify({"msg": "success"}), 200
                    else:
                        return jsonify({"msg": self.response_start_collection['msg']}), 404
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 5: # 正式环境设为2.5超时
                    return jsonify({"msg": "机器人响应超时"}), 404

        except Exception as e:
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def finish_collection(self):
        try:
            data = request.get_json()
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'finish_collection'})
            while True:
                if 0 < self.response_finish_collection["timestamp"] - now_time < 3:
                    if self.response_finish_collection['msg'] == "success":
                        return jsonify(self.response_finish_collection['data']), 200
                    else:
                        return jsonify({"msg": self.response_finish_collection['msg']}), 404
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 5: # 正式环境设为2.5超时
                    return jsonify({"msg": "机器人响应超时"}), 404

        except Exception as e:
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
        
    def discard_collection(self):
        try:
            data = request.get_json()
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'discard_collection'})
            while True:
                if 0 < self.response_discard_collection["timestamp"] - now_time < 3:
                    if self.response_discard_collection['msg'] == "success":
                        return jsonify({"msg": "success"}), 200
                    else:
                        return jsonify({"msg": self.response_discard_collection['msg']}), 404
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 5: # 正式环境设为2.5超时
                    return jsonify({"msg": "机器人响应超时"}), 404

        except Exception as e:
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
        
    def submit_collection(self):
        try:
            data = request.get_json()
            now_time = time.time()
            self.send_message_to_robot(self.robot_sid, message={'cmd': 'submit_collection'})
            while True:
                if 0 < self.response_submit_collection["timestamp"] - now_time < 3:
                    if self.response_submit_collection['msg'] == "success":
                        return jsonify({"msg": "success"}), 200
                    else:
                        return jsonify({"msg": self.response_submit_collection['msg']}), 404
                else:
                    time.sleep(0.02)
                if time.time() - now_time > 5: # 正式环境设为2.5超时
                    return jsonify({"msg": "机器人响应超时"}), 404

        except Exception as e:
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def standby(self):
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        pass


    # ---------------------------------------robot------------------------------------------------
    def update_frame(self, stream_id):
        try:
            # 尝试将 stream_id 转换为整数
            stream_id = int(stream_id)
        except ValueError:
            return jsonify({"error": "update_frame ID, must be int"}), 400

        if stream_id not in self.video_streams:
            logging.error(f"Invalid stream ID: {stream_id}")
            return jsonify({"error": "Invalid stream ID"}), 401

        frame_data = request.get_data()
        if not frame_data:
            logging.error("No frame data received")
            return jsonify({"error": "didn't recieved data"}), 402
        
        try:
            # 可选：验证是否为 JPEG 数据
            img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Invalid JPEG data")
        except Exception as e:
            logging.error(f"Invalid frame data: {e}")
            return jsonify({"error": "Invalid frame data"}), 500

        self.video_streams[stream_id].update_frame(frame_data)
        return jsonify({"msg": "帧已更新"}), 200
    
    def robot_get_video_list(self):
        try:
            new_list = request.get_json()
            self.video_list = json.loads(new_list)
            self.video_timestamp = time.time()
            if not self.init_streams_flag:
                self.init_streams()
                self.init_streams_flag = True
            return jsonify({}), 200
        except Exception as e:
            print(f"error: {e}")
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def robot_response(self):
        try:
            data = request.get_json()
            if data["cmd"] == "start_collection":
                self.response_start_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"]
                }
            elif data["cmd"] == "finish_collection":
                self.response_finish_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"],
                    "data": data["data"]
                }
            elif data["cmd"] == "discard_collection":
                self.response_discard_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"]
                }
            elif data["cmd"] == "submit_collection":
                self.response_submit_collection = {
                    "timestamp": time.time(),
                    "msg": data["msg"]
                }
            return jsonify({}), 200
        except Exception as e:
            return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    def get_task_steps(self):
        return jsonify(self.task_steps), 200
    
    def run(self):
        """运行服务器"""
        self.socketio.run(self.app, host='0.0.0.0', port=8080, debug=True)



if __name__ == '__main__':
    server = FlaskServer()
    server.run()