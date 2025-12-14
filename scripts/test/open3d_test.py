import open3d as o3d
import numpy as np

# 创建一个简单球体
mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([1, 0, 0])  # 红色，更明显

# 方法1: 传统 API
# o3d.visualization.draw_geometries([mesh])

# 方法2: 新版 draw API (0.18+)，渲染更稳定
o3d.visualization.draw([mesh], title="Open3D Test")