# roop-player
实时视频换脸播放器
支持视频、图片实时换脸预览，支持对应用目标窗口换脸。只支持画面无声音，因为画面延迟声音无法同步，也无法保存视频。实验娱乐项目，切勿非法用途。

## 运行环境与需要的模型与roop要求相同，可先尝试搭建roop虚拟环境
## 注意 onnxruntime 推理库版本，有cpu、cuda11和cuda12区别，详情请看onnxruntime文档。建议安装 onnxruntime-gpu 版本（默认cuda12）。
gfpgan依赖bug修复：
"E:\Github\venv\Lib\site-packages\basicsr\data\degradations.py" 第8行引用可能会报错： from torchvision.transforms.functional_tensor import rgb_to_grayscale
修改为 from torchvision.transforms._functional_tensor import rgb_to_grayscale
