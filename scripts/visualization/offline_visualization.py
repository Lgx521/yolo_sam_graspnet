"""
离线可视化脚本.
功能:
- 读取由 'run_and_save_graspnet.py' 保存的结果目录。
- 加载 'scene.pcd' 点云文件。
- 加载 'grasp_results.mat' 文件，其中包含抓取姿态、分数和宽度。
- 将抓取数据转换为GraspGroup对象。
- 使用Open3D可视化点云和抓取指示器。
"""

import numpy as np
import open3d as o3d
import scipy.io as scio
import argparse
import os
import sys

# --- 确保GraspNet相关模块在PYTHONPATH中 ---
try:
    from graspnetAPI import GraspGroup
except ImportError:
    print("\n错误：无法导入 graspnetAPI。")
    print("请确保 graspnet-baseline 的根目录在您的 PYTHONPATH 环境变量中。")
    print("例如: export PYTHONPATH=$PYTHONPATH:/path/to/your/graspnet-baseline\n")
    sys.exit(1)


def load_and_convert_grasps(mat_path):
    """
    从 .mat 文件加载抓取数据并将其转换为 GraspGroup 对象。
    
    Args:
        mat_path (str): .mat 文件的路径。

    Returns:
        graspnetAPI.GraspGroup: 包含所有已加载抓取的 GraspGroup 对象。
    """
    try:
        data = scio.loadmat(mat_path)
    except FileNotFoundError:
        print(f"错误: .mat 文件未找到于 '{mat_path}'")
        return None

    # 从.mat文件中提取数据
    if 'poses' not in data or 'scores' not in data or 'widths' not in data:
        print(f"错误: .mat 文件 '{mat_path}' 缺少 'poses', 'scores', 或 'widths' 键。")
        return None

    poses = data['poses']
    scores = data['scores'].flatten() # 确保是一维数组
    widths = data['widths'].flatten() # 确保是一维数组

    num_grasps = poses.shape[2]
    if not (len(scores) == num_grasps and len(widths) == num_grasps):
        print("警告: poses, scores, 和 widths 的数量不匹配。")
        return None

    gg_array = []
    for i in range(num_grasps):
        # 提取单个 3x4 变换矩阵
        pose = poses[:, :, i]
        
        # 提取旋转矩阵和平移向量
        rotation_matrix = pose[:3, :3]
        translation = pose[:3, 3]
        
        # 获取对应的分数和宽度
        score = scores[i]
        width = widths[i]

        # 构建 GraspGroup 所需的17元素行
        single_grasp_row = np.zeros(17)
        single_grasp_row[0] = score
        single_grasp_row[1] = width
        single_grasp_row[2] = 0.04  # 默认抓取器高度
        single_grasp_row[3] = 0.0   # 默认抓取深度
        single_grasp_row[4:13] = rotation_matrix.flatten()
        single_grasp_row[13:16] = translation
        single_grasp_row[16] = -1   # 默认物体ID
        gg_array.append(single_grasp_row)

    if not gg_array:
        return GraspGroup() # 返回一个空的GraspGroup

    return GraspGroup(np.array(gg_array))


def main():
    parser = argparse.ArgumentParser(description="离线可视化GraspNet结果")
    parser.add_argument('--top_k', type=int, default=50,
                        help="最多显示前 K 个最佳抓取。")
    args = parser.parse_args()

    # 1. 构建文件路径
    script_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(script_path)

    pcd_path= folder_path + '/scene.pcd'
    mat_path= folder_path + '/grasp_results.mat'

    # 2. 检查文件是否存在
    if not os.path.exists(pcd_path) or not os.path.exists(mat_path):
        # print(f"\n错误: 在目录 '{args.results_dir}' 中未找到所需文件。")
        print(f"请确保 '{pcd_path}' 和 '{mat_path}' 都存在。\n")
        sys.exit(1)

    # 3. 加载点云
    print(f"正在加载点云: {pcd_path}")
    cloud = o3d.io.read_point_cloud(pcd_path)
    if not cloud.has_points():
        print("错误: 加载的点云为空。")
        sys.exit(1)

    # 4. 加载并转换抓取
    print(f"正在加载抓取结果: {mat_path}")
    gg = load_and_convert_grasps(mat_path)
    if gg is None:
        sys.exit(1)
        
    if len(gg) == 0:
        print("未加载到任何抓取，只显示点云。")
        o3d.visualization.draw_geometries([cloud])
        return

    print(f"成功加载 {len(gg)} 个抓取。正在准备可视化...")

    # 5. 准备可视化
    # 按分数排序并筛选前K个
    gg.sort_by_score()
    gg_top_k = gg[:args.top_k]
    
    # 获取抓取器的3D几何模型列表
    grippers = gg_top_k.to_open3d_geometry_list()

    # 6. 启动可视化
    print(f"正在显示点云和前 {len(grippers)} 个最佳抓取...")
    o3d.visualization.draw_geometries(
        [cloud, *grippers],
        window_name=f"GraspNet Visualization "
    )

if __name__ == '__main__':
    main()