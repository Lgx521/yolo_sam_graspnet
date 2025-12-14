# GraspNet 处理与可视化脚本

该项目包含两个核心的 Python 脚本，旨在提供一个完整的流程：从实时ROS2相机数据运行GraspNet，到离线回顾和分析检测结果。

## 核心脚本

1.  **`run_and_save_graspnet.py`**: 一个在线的、一站式的处理脚本。
2.  **`visualize_saved_results.py`**: 一个离线的、纯粹的可视化脚本。

---

## 依赖与准备

在使用这些脚本前，请确保您已准备好：

*   一个配置完善的 **GraspNet** Python 环境 (PyTorch, Open3D, SciPy, 等)。
*   一个正在运行的 **ROS2 Humble** 环境。
*   一个正在发布数据的 **RGB-D 相机**，其话题应包含：
    *   `/camera/color/image_raw`
    *   `/camera/depth/image_raw`
    *   `/camera/color/camera_info`
*   一个已下载的 GraspNet **预训练模型检查点** (例如 `checkpoint-rs.tar`)。

---

## 脚本详解

### 1. `visualization.py` - 实时处理与保存

这个是**主脚本**，用于执行从数据捕获到结果保存的全过程。

#### **功能**

*   连接到实时ROS2话题。
*   捕获一帧同步的图像和点云数据。
*   运行GraspNet模型进行抓取姿态推理。
*   将结果保存到指定目录。
*   启动一个即时预览窗口。

#### **如何使用**

在终端中运行以下命令。您必须提供模型检查点的路径。

```bash
python run_and_save_graspnet.py --checkpoint_path /path/to/your/checkpoint-rs.tar
```

您也可以指定一个输出目录：

```bash
python run_and_save_graspnet.py --checkpoint_path /path/to/your/checkpoint-rs.tar --output_dir ./my_capture_01
```

#### **输出**

该脚本会创建一个输出目录（默认为 `graspnet_ros_output`），其中包含：

*   `scene.pcd`: 在该场景中检测时使用的点云。
*   `grasp_results.mat`: 一个包含所有抓取姿态、分数和宽度的MATLAB文件。

### 2. `offline_visualization.py` - 离线结果可视化

当您想**回顾或演示**之前保存的结果时，使用此脚本。它**不需要ROS或神经网络**，运行速度非常快。

#### **功能**

*   读取由 `run_and_save_graspnet.py` 生成的结果目录。
*   加载点云和抓取姿态数据。
*   在Open3D窗口中显示它们。

#### **如何使用**

Copy this script to specific output folder which contains `grasp_results.mat` and `scene.pcd`  
Then run this script.

您可以指定只显示分数最高的前K个抓取：

```bash
python visualize_saved_results.py ./my_capture_01 --top_k 20
```

---

## 推荐工作流程

1.  **数据采集与处理**: 当您的相机和场景就位后，运行 `run_and_save_graspnet.py` 来捕获场景，运行模型，并获得即时预览。
2.  **回顾与分析**: 在任何时候，您都可以使用 `visualize_saved_results.py` 来快速加载任何一次的捕获结果，以便进行详细检查、截图或演示，而无需重新运行整个耗时的推理过程。
