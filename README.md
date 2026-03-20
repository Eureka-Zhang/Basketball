# 篮球投篮检测与跟踪系统

## 项目简介
这是一个基于YOLOv8模型的篮球投篮检测与跟踪系统。该项目旨在通过视频分析自动检测篮球的投篮尝试和命中情况，并提供实时反馈。

## 功能特点
- **投篮检测**：识别篮球和篮球框的位置。
- **命中判断**：判断投篮是否命中。
- **实时反馈**：在视频中实时显示投篮结果（命中或未命中）。
- **统计数据**：记录投篮尝试次数和命中次数，并计算命中率。

## 文件结构
```
.
├── README.md                # 项目说明文件
├── test_video/              # 测试视频存放目录
├── yolov8+Interpolation/    # 主程序目录
│   ├── best.pt              # YOLOv8模型权重文件
│   ├── requirements.txt     # Python依赖包列表
│   ├── shot_detector.py     # 主程序代码
│   └── utils.py             # 辅助功能模块
```

## 环境配置
1. 克隆仓库：
   ```bash
   git clone <仓库地址>
   ```
2. 进入项目目录：
   ```bash
   cd yolov8+Interpolation
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法
1. 将测试视频放入`test_video`目录。
2. 运行主程序：
   ```bash
   python shot_detector.py
   ```
3. 按`q`键退出程序。

## 注意事项
- 请确保`best.pt`模型文件存在于`yolov8+Interpolation`目录下。
- 测试视频的路径应为`test_video/clip3_shoot.mp4`，或根据需要修改代码中的路径。

## 贡献
欢迎提交问题（Issues）或拉取请求（Pull Requests）以改进本项目。

## 许可证
本项目遵循MIT许可证。