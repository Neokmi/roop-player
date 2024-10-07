import glob
import os
import time
from typing import Any, List, Optional, Tuple
from threading import Thread
import cv2
import insightface
from PIL.Image import Image
from cv2 import VideoCapture, Mat
from insightface.app.common import Face
from gfpgan.utils import GFPGANer
import customtkinter as ctk
from numpy import ndarray, dtype
from tkinterdnd2 import TkinterDnD, DND_ALL
from PIL import Image, ImageOps
import mss
import numpy as np
import pygetwindow as gw
import mimetypes
import mss
import pygetwindow as gw
from PIL import Image



Frame = np.ndarray[Any, Any]

class CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

class VideoPlayerApp:
    def __init__(self):
        # 初始化视频播放相关变量
        # 源
        self.RECENT_DIRECTORY_SOURCE = "h:/"
        self.src_path = self.RECENT_DIRECTORY_SOURCE
        self.src_path_list = []
        self.src_id = 0
        self.src_cap = None
        self.src_frame_count = 0
        self.update_src_bars = 0
        # 目标
        self.RECENT_DIRECTORY_TARGET = "h:/"
        self.target_path = self.RECENT_DIRECTORY_TARGET
        self.target_path_list = []
        self.target_id = 0
        self.target_cap = None
        self.target_frame_count = 0
        self.update_target_bars = 0
        # 缓存源人脸
        self.face_list = []
        self.face_id = 0
        self.target_frame = None
        self.mode = ""
        # 播放器参数
        self.video_extensions = ("mp4", "avi", "mkv", "rm", "rmvb", "3gp", "mov", "mpeg", "mpg", "wmv")
        self.image_extensions = ("jpg", "jpeg", "bmp", "png", "webp")
        # 进度条参数
        self.delay_ms = 0
        self.frame_step = 1
        self.zoom = 1
        self.enhance = 0
        self.play_state = False
        # 加载模型
        execution_providers = ["CUDAExecutionProvider"]
        self.FACE_SWAPPER = insightface.model_zoo.get_model(r'models\inswapper_128.onnx', providers=execution_providers)
        self.FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=execution_providers)
        self.FACE_ANALYSER.prepare(ctx_id=0)
        self.FACE_ENHANCER = GFPGANer(model_path=r'models\GFPGANv1.4.pth', upscale=1, device='cuda')
        # 播放窗口
        self.ROOT_HEIGHT = 600
        self.ROOT_WIDTH = 500
        ctk.deactivate_automatic_dpi_awareness()
        ctk.set_appearance_mode('system')
        # ctk.set_default_color_theme(resolve_relative_path('ui.json'))
        root = CTk()
        root.minsize(self.ROOT_WIDTH, self.ROOT_HEIGHT)
        root.title(f'Roop-Player')
        root.configure()
        root.protocol('WM_DELETE_WINDOW', lambda: exit())
        self.source_label = ctk.CTkLabel(root, text="")
        self.source_label.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.25)
        self.source_label.drop_target_register(DND_ALL)
        self.source_label.dnd_bind('<<Drop>>', lambda event: self.set_source_img_box(event.data))
        self.target_label = ctk.CTkLabel(root, text="")
        self.target_label.place(relx=0.6, rely=0.1, relwidth=0.3, relheight=0.25)
        self.target_label.drop_target_register(DND_ALL)
        self.target_label.dnd_bind('<<Drop>>', lambda event: self.set_target_img_box(event.data))
        self.source_button = ctk.CTkButton(root, text='Select a src', cursor='hand2',command=lambda: self.select_source_path())
        self.source_button.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)
        self.target_button = ctk.CTkButton(root, text='Select a target', cursor='hand2',command=lambda: self.select_target_path())
        self.target_button.place(relx=0.6, rely=0.4, relwidth=0.3, relheight=0.1)
        self.source_button = ctk.CTkButton(root, text='View_source', cursor='hand2',command=lambda: self.view_file(self.src_path))
        self.source_button.place(relx=0.1, rely=0.6, relwidth=0.3, relheight=0.1)
        self.target_button = ctk.CTkButton(root, text='View_target', cursor='hand2',command=lambda:  self.view_file(self.target_path))
        self.target_button.place(relx=0.6, rely=0.6, relwidth=0.3, relheight=0.1)
        # self.keep_frames_value = ctk.BooleanVar(value=True)
        # self.keep_frames_switch = ctk.CTkSwitch(root, text='Keep temporary frames', variable=None,cursor='hand2')
        # self.keep_frames_switch.place(relx=0.1, rely=0.65)
        # self.skip_audio_value = ctk.BooleanVar(value=roop.globals.skip_audio)
        # self.skip_audio_switch = ctk.CTkSwitch(root, text='Skip target audio', variable=skip_audio_value, cursor='hand2',command=lambda: setattr(roop.globals, 'skip_audio', skip_audio_value.get()))
        # self.skip_audio_switch.place(relx=0.6, rely=0.6)
        self.start_button = ctk.CTkButton(root, text='Rec_src', cursor='hand2', command=lambda: self.play_new_Thread("src"))
        self.start_button.place(relx=0.15, rely=0.75, relwidth=0.2, relheight=0.05)
        self.stop_button = ctk.CTkButton(root, text='Rec_target', cursor='hand2', command=lambda:self.play_new_Thread("target"))
        self.stop_button.place(relx=0.4, rely=0.75, relwidth=0.2, relheight=0.05)
        # # 创建播放按钮
        self.preview_button = ctk.CTkButton(root, text='Preview', cursor='hand2', command=lambda: self.play_new_Thread())
        self.preview_button.place(relx=0.65, rely=0.75, relwidth=0.2, relheight=0.05)
        # self.status_label = ctk.CTkLabel(root, text=None, justify='center')
        # self.status_label.place(relx=0.1, rely=0.9, relwidth=0.8)
        # 启动窗口
        root.mainloop()
    def view_file(self,path):
        path=path.replace("/","\\")
        cmd = f'explorer /select, "{path}"'
        os.system(cmd)
    def is_image(self,image_path: str) -> bool:
        if image_path and os.path.isfile(image_path):
            if image_path.endswith(self.image_extensions):
                return True
            mimetype, _ = mimetypes.guess_type(image_path)
            return bool(mimetype and mimetype.startswith('image/'))
        return False
    def is_video(self,video_path: str) -> bool:
        if video_path and os.path.isfile(video_path):
            if video_path.endswith(self.video_extensions):
                return True
            mimetype, _ = mimetypes.guess_type(video_path)
            return bool(mimetype and mimetype.startswith('video/'))
        return False
    def render_image_preview(self,image_path, size,label_box) -> ndarray[Any, dtype[Any]]:
        if isinstance(image_path,str):
            if os.path.isfile(image_path):
                image = Image.open(image_path)
        else:
            image = image_path
        if size:
            fit_image = ImageOps.fit(image, size, Image.LANCZOS)
        else:
            fit_image = image
        label_box.configure(image=ctk.CTkImage(fit_image, size=fit_image.size))
        return cv2.cvtColor(np.array(image,dtype=np.uint8), cv2.COLOR_RGB2BGR)
    def render_video_preview(self,video_path: str, size: Tuple[int, int],label_box, frame_number: int = 0) -> tuple[Mat | ndarray, VideoCapture]:
        capture = cv2.VideoCapture(video_path)
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if has_frame:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if size:
                image = ImageOps.fit(image, size, Image.LANCZOS)
            label_box.configure(image=ctk.CTkImage(image, size=image.size))
            return frame, capture
        capture.release()
        cv2.destroyAllWindows()
        return None, None
    def refresh_source(self,src_path):
        # 显示图片
        if self.is_image(src_path):
            image, self.src_cap = self.render_image_preview(src_path, (200, 200),self.source_label),None
            self.src_frame_count = 0
        elif self.is_video(src_path):
            image, self.src_cap = self.render_video_preview(src_path, (200, 200),self.source_label)
            self.src_frame_count = int(self.src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            return
        # 更新人脸
        self.face_list = self.get_many_faces(image)
    def refresh_target(self,target_path):
        # 显示图片
        if self.is_image(target_path):
            first_frame, self.target_cap = self.render_image_preview(target_path, (200, 200),self.target_label), None
            self.target_frame_count = 0
        elif self.is_video(target_path):
            first_frame, self.target_cap = self.render_video_preview(target_path, (200, 200),self.target_label)
            self.target_frame_count = int(self.target_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            return None
        return first_frame
    def select_source_path(self):
        src_path  = ctk.filedialog.askopenfilename(title='select an source image',initialdir=self.RECENT_DIRECTORY_SOURCE)
        if src_path:
            # 初始化目录及文件
            self.RECENT_DIRECTORY_SOURCE = os.path.dirname(src_path)
            self.src_path_list = [src_path]
            for ext in (self.video_extensions + self.image_extensions):
                self.src_path_list.extend(glob.glob(f"{self.RECENT_DIRECTORY_SOURCE}/*.{ext}"))
            self.src_path = src_path
            self.refresh_source(self.src_path)
            self.update_src_bars = 1
    def select_target_path(self):
        target_path = ctk.filedialog.askopenfilename(title='select an target image or video', initialdir=self.RECENT_DIRECTORY_TARGET) #,filetypes=[("target files", "*.mp4;*.avi;*.mkv")]
        if target_path:
            # 初始化目录及文件
            self.RECENT_DIRECTORY_TARGET = os.path.dirname(target_path)
            self.target_path_list = [target_path]
            for ext in (self.video_extensions + self.image_extensions):
                self.target_path_list.extend(glob.glob(f"{self.RECENT_DIRECTORY_TARGET}/*.{ext}"))
            self.target_path = target_path
            self.refresh_target(self.target_path)
            self.update_target_bars = 1
    def set_target_path(self,v):
        if v >= len(self.target_path_list):
            return
        self.target_path = self.target_path_list[v]
        self.refresh_target(self.target_path)
        cv2.setTrackbarPos("target_frame", "cmd", 0)
        cv2.setTrackbarMax("target_frame", "cmd", self.target_frame_count)
    def set_src_path(self, v):
        if v >= len(self.src_path_list):
            return
        self.src_path = self.src_path_list[v]
        self.refresh_source(self.src_path)
        cv2.setTrackbarPos("src_frame", "cmd", 0)
        cv2.setTrackbarMax("src_frame", "cmd", self.src_frame_count)
        cv2.setTrackbarPos("face_id", "cmd", 0)
        cv2.setTrackbarMax("face_id", "cmd", len(self.face_list)-1)
    def set_face_id(self, v):
        if v > len(self.face_list):
            return
        self.face_id = v
    def set_delay(self, v):
        self.delay_ms = v
    def set_zoom(self, v):
        self.zoom = v
    def set_step(self, v):
        self.frame_step = v
    def set_enhance(self,v):
        self.enhance = True if v > 0 else False
    def set_src_frame(self,v):
        if self.src_cap is None:
            return
        if v >= self.src_frame_count:
            return
        self.src_cap.set(cv2.CAP_PROP_POS_FRAMES, v)
        has_frame, frame = self.src_cap.read()
        size=(200, 200)
        if has_frame:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if size:
                image = ImageOps.fit(image, size, Image.LANCZOS)
            self.source_label.configure(image=ctk.CTkImage(image, size=image.size))
            self.face_list = self.get_many_faces(frame)
            cv2.setTrackbarPos("face_id", "cmd", 0)
            cv2.setTrackbarMax("face_id", "cmd", len(self.face_list)-1)
    def set_target_frame(self, v):
        if self.target_cap is None:
            return
        if v >= self.target_frame_count:
            return
        self.target_cap.set(cv2.CAP_PROP_POS_FRAMES, v)
    def play_new_Thread(self,cmd="play"):
        self.play_state=True
        if cmd=="play":
            self.mode = ""
            Thread(target=self.play_video).start()
        elif cmd=="src":
            self.mode=""
            Thread(target=self.rec_src).start()
            time.sleep(1)
            Thread(target=self.play_video).start()
        elif cmd=="target":
            self.mode = "target"
            Thread(target=self.rec_target).start()
            time.sleep(1)
            Thread(target=self.play_video).start()
    def int_srcList_trackbar(self):
        if cv2.getWindowProperty("cmd",cv2.WND_PROP_VISIBLE)>=1:
            cv2.setTrackbarPos("src", "cmd", 0)
            cv2.setTrackbarMax("src", "cmd", len(self.src_path_list))
            cv2.setTrackbarPos("face_id", "cmd", 0)
            cv2.setTrackbarMax("face_id", "cmd", len(self.face_list)-1)
            cv2.setTrackbarPos("src_frame", "cmd", 0)
            cv2.setTrackbarMax("src_frame", "cmd", self.src_frame_count - 1)
        self.update_src_bars = 0
    def int_targetList_trackbar(self):
        if cv2.getWindowProperty("cmd", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.setTrackbarPos("target", "cmd", 0)
            cv2.setTrackbarMax("target", "cmd", len(self.target_path_list))
            cv2.setTrackbarPos("target_frame", "cmd", 0)
            cv2.setTrackbarMax("target_frame", "cmd", self.target_frame_count -1)
        self.update_target_bars = 0
    def play_video(self):
        #self.play_button.config(state=tk.DISABLED)
        cv2.namedWindow('Video Player',0)
        cv2.namedWindow('cmd',0)
        cv2.createTrackbar("target", "cmd", 0, len(self.target_path_list) - 1, self.set_target_path)
        cv2.createTrackbar("target_frame","cmd",0,self.target_frame_count -1,self.set_target_frame)
        cv2.createTrackbar("src", "cmd", 0, len(self.src_path_list) - 1, self.set_src_path)
        cv2.createTrackbar("src_frame", "cmd", 0, self.src_frame_count - 1, self.set_src_frame)
        cv2.createTrackbar("face_id", "cmd", 0, len(self.face_list)-1 if self.face_list else 10, self.set_face_id)
        cv2.createTrackbar("enhance", "cmd", 0, 1, self.set_enhance)
        cv2.createTrackbar("delay", "cmd", 0, 500, self.set_delay)
        cv2.createTrackbar("step", "cmd", 1, 25, self.set_step)
        cv2.createTrackbar("zoom", "cmd", 5, 30, self.set_zoom)
        cv2.resizeWindow("cmd",300,400)
        self.play_state=True

        while 1:
            if self.update_src_bars == 1:
                self.int_srcList_trackbar()
            if self.update_target_bars == 1:
                self.int_targetList_trackbar()
            if self.mode!="target":
                if self.is_video(self.target_path):
                    # 更新帧进度
                    current_frame = cv2.getTrackbarPos("target_frame", "cmd") + self.frame_step
                    if current_frame > self.target_frame_count-1:
                        current_frame = self.target_frame_count-1
                    self.target_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    cv2.setTrackbarPos("target_frame", "cmd", current_frame)
                    # 读取视频帧
                    ret, self.target_frame = self.target_cap.read()
                    if not ret:
                        video_id = cv2.getTrackbarPos('target', 'cmd') + 1
                        if video_id > len(self.target_path_list) - 1:
                            video_id = 0
                        cv2.setTrackbarPos("target", "cmd", video_id)
                elif self.is_image(self.target_path):
                    self.target_frame = self.refresh_target(self.target_path)
            if self.target_frame is not None:
                # 处理帧
                frame = self.process_frame(self.target_frame)
                # 显示帧
                cv2.imshow('Video Player', frame)

                # 按下 q 键退出循环
                if cv2.waitKey(self.delay_ms + 1) & 0xFF == ord('q') or cv2.getWindowProperty("cmd",cv2.WND_PROP_VISIBLE)<1 or cv2.getWindowProperty("Video Player",cv2.WND_PROP_VISIBLE)<1:
                    self.play_state=False
                    break
            # 每帧间隔时间（单位：毫秒）
            # interval = int(1000 / self.cap.get(cv2.CAP_PROP_FPS))
            # time.sleep(interval / 1000)

        # 释放资源
        # if self.src_cap and self.src_cap.isOpened():
        #     self.src_cap.release()
        cv2.destroyAllWindows()
        self.play_state = False
    def get_many_faces(self, frame: Frame) -> Optional[List[Face]]:
        try:
            j=self.FACE_ANALYSER.get(frame,max_num=10)
            return j
        except Exception as e:
            print(e)
            return None
    def enhance_face(self, target_face: Face, temp_frame: Frame) -> Frame:
        start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
        padding_x = int((end_x - start_x) * 0.5)
        padding_y = int((end_y - start_y) * 0.5)
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = max(0, end_x + padding_x)
        end_y = max(0, end_y + padding_y)
        temp_face = temp_frame[start_y:end_y, start_x:end_x]
        if temp_face.size:
            _, _, temp_face = self.FACE_ENHANCER.enhance(temp_face,paste_back=True)
            temp_frame[start_y:end_y, start_x:end_x] = temp_face
        return temp_frame
    def process_frame(self, temp_frame: Frame) -> Frame:
        temp_frame = self.target_frame
        n=5
        many_faces = self.get_many_faces(temp_frame)
        if many_faces and self.face_list and len(self.face_list) >= 1 and len(self.face_list) > self.face_id:
            for target_face in many_faces:
                if self.face_list and len(self.face_list) >= 1 and len(self.face_list) > self.face_id:
                    temp_frame = self.FACE_SWAPPER.get(temp_frame, target_face, self.face_list[self.face_id], paste_back=True)
                    if self.enhance:
                        temp_frame = self.enhance_face(target_face, temp_frame)
        h, w = temp_frame.shape[:2]
        cv2.resizeWindow('Video Player', int(w * self.zoom / n), int(h * self.zoom / n))
        return temp_frame

    def get_win_size(self):
        title_keyword = "zy"
        try:
            all_windows = gw.getWindowsWithTitle('zy')
            matching_windows = []
            for w in all_windows:
                # print(w.title.lower())
                if title_keyword.lower() in w.title.lower():
                    matching_windows.append(w)
            if not matching_windows:
                print("no win")
                return (100, 100, 100, 100)
            window = matching_windows[0]
            left, top, width, height = window.left, window.top, window.width, window.height
            return (100, 100, 100, 100) if (width < 50 or height < 50 or left < 0 or top < 0) else (left, top, width, height)
        except Exception as e:
            print("dddd: ",e)
        return (100, 100, 100, 100)

    def rec_src(self):
        while self.play_state:
            with (mss.mss() as sct):
                screenshot = sct.grab(sct.monitors[1])  # 默认显示器
                numpy_image = np.array(screenshot, dtype=np.uint8)
                left, top, width, height = self.get_win_size()
                cropped_image = numpy_image[top:top + height, left:left + width]
                if cropped_image.shape[2] == 4:
                    cropped_image = cropped_image[:, :, :3]  # 只保留前三个通道 (R, G, B)
                pil_image =Image.fromarray(cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB))
                self.render_image_preview(pil_image, (200, 200), self.source_label)
                self.face_list = self.get_many_faces(cropped_image)
                # time.sleep(0.5)
    def rec_target(self):
        while self.play_state:
            with mss.mss() as sct:
                screenshot = sct.grab(sct.monitors[1])  # 默认显示器
                numpy_image = np.array(screenshot)
                left, top, width, height = self.get_win_size()
                cropped_image = numpy_image[top:top + height, left:left + width]
                if cropped_image.shape[2] == 4:
                    cropped_image = cropped_image[:, :, :3]  # 只保留前三个通道 (R, G, B)
               # pil_image =Image.fromarray(cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB))
                #self.render_image_preview(pil_image,(200,200),self.target_label)
                self.target_frame = cropped_image
                # time.sleep(0.5)


if __name__ == '__main__':
    app = VideoPlayerApp()
