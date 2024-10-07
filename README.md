# roop-player
实时视频换脸播放器

## 运行环境与需要的模型与roop要求相同，可先尝试搭建roop虚拟环境
## 注意 onnxruntime 推理库版本，有cpu、cuda11和cuda12区别，详情请看onnxruntime文档。建议安装 onnxruntime-gpu 版本（默认cuda12）。
gfpgan依赖bug修复：
"E:\Github\venv\Lib\site-packages\basicsr\data\degradations.py" 第8行引用可能会报错： from torchvision.transforms.functional_tensor import rgb_to_grayscale
修改为 from torchvision.transforms._functional_tensor import rgb_to_grayscale
