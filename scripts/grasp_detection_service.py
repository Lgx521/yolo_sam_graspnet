#!/usr/bin/env python3

import os
import sys
# Add graspnet-baseline to Python path
sys.path.insert(0, '/home/roar/graspnet/graspnet-baseline')
sys.path.insert(0, '/home/roar/graspnet/graspnet-baseline/models')
sys.path.insert(0, '/home/roar/graspnet/graspnet-baseline/utils')
import numpy as np
import torch
import time
from typing import Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# Add GraspNet paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from models.graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo as GraspNetCameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup
import open3d as o3d

# Import custom service definitions
from kinova_graspnet_ros2.srv import DetectGrasps
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, Point, Quaternion


class GraspDetectionService(Node):
    """ROS2 service node for GraspNet-based grasp detection"""
    
    def __init__(self):
        super().__init__('grasp_detection_service')
        
        # Declare parameters
        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('num_point', 25000)
        self.declare_parameter('collision_thresh', 0.01)
        self.declare_parameter('voxel_size', 0.01)
        self.declare_parameter('device', 'cuda:0')
        
        # RGB Camera parameters (1920x1080)
        self.declare_parameter('rgb_camera_intrinsics.fx', 2612.345947)
        self.declare_parameter('rgb_camera_intrinsics.fy', 2609.465576)
        self.declare_parameter('rgb_camera_intrinsics.cx', 946.433838)
        self.declare_parameter('rgb_camera_intrinsics.cy', 96.968155)
        self.declare_parameter('rgb_camera_intrinsics.width', 1920)
        self.declare_parameter('rgb_camera_intrinsics.height', 1080)
        
        # Depth Camera parameters (480x270)
        self.declare_parameter('depth_camera_intrinsics.fx', 336.190430)
        self.declare_parameter('depth_camera_intrinsics.fy', 336.190430)
        self.declare_parameter('depth_camera_intrinsics.cx', 230.193512)
        self.declare_parameter('depth_camera_intrinsics.cy', 138.111130)
        self.declare_parameter('depth_camera_intrinsics.width', 480)
        self.declare_parameter('depth_camera_intrinsics.height', 270)
        
        # Target object class for segmentation
        self.declare_parameter('target_object_class', '')
        
        # Frame configuration
        self.declare_parameter('rgb_camera_frame', 'camera_color_frame')
        self.declare_parameter('depth_camera_frame', 'camera_depth_frame')
        self.declare_parameter('base_frame', 'base_link')
        
        # Camera topic configuration
        self.declare_parameter('color_image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_image_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        
        # Get parameters
        self.checkpoint_path = self.get_parameter('checkpoint_path').value
        self.num_point = self.get_parameter('num_point').value
        self.collision_thresh = self.get_parameter('collision_thresh').value
        self.voxel_size = self.get_parameter('voxel_size').value
        device_name = self.get_parameter('device').value
        
        # Get RGB camera parameters
        self.rgb_camera_params = {
            'fx': self.get_parameter('rgb_camera_intrinsics.fx').value,
            'fy': self.get_parameter('rgb_camera_intrinsics.fy').value,
            'cx': self.get_parameter('rgb_camera_intrinsics.cx').value,
            'cy': self.get_parameter('rgb_camera_intrinsics.cy').value,
            'width': self.get_parameter('rgb_camera_intrinsics.width').value,
            'height': self.get_parameter('rgb_camera_intrinsics.height').value
        }
        
        # Get depth camera parameters
        self.depth_camera_params = {
            'fx': self.get_parameter('depth_camera_intrinsics.fx').value,
            'fy': self.get_parameter('depth_camera_intrinsics.fy').value,
            'cx': self.get_parameter('depth_camera_intrinsics.cx').value,
            'cy': self.get_parameter('depth_camera_intrinsics.cy').value,
            'width': self.get_parameter('depth_camera_intrinsics.width').value,
            'height': self.get_parameter('depth_camera_intrinsics.height').value
        }
        
        # Get target object class
        self.default_target_object_class = self.get_parameter('target_object_class').value
        
        # Get frame names
        self.rgb_camera_frame = self.get_parameter('rgb_camera_frame').value
        self.depth_camera_frame = self.get_parameter('depth_camera_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        
        # Get camera topics
        self.color_image_topic = self.get_parameter('color_image_topic').value
        self.depth_image_topic = self.get_parameter('depth_image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        
        # Initialize device
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Image data storage
        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_camera_info = None
        self.images_ready = False
        
        # Camera subscribers
        self.color_sub = self.create_subscription(
            Image,
            self.color_image_topic,
            self.color_image_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_image_topic,
            self.depth_image_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        
        
        # Load GraspNet model
        self.net = self._load_model()
        
        # Create service
        self.srv = self.create_service(
            DetectGrasps,
            'detect_grasps',
            self.detect_grasps_callback
        )
        
        # Initialize visualization publishers
        self.grasp_marker_pub = self.create_publisher(
            MarkerArray,
            'grasp_markers',
            10
        )
        
        self.debug_image_pub = self.create_publisher(
            Image,
            'debug_grasp_image',
            10
        )
        
        # Publisher for YOLO detection visualization
        self.yolo_detection_pub = self.create_publisher(
            Image,
            'yolo_detection_visualization',
            10
        )
        
        self.get_logger().info(
            f'GraspNet detection service initialized. Using device: {self.device}'
        )
        self.get_logger().info(f'Subscribing to camera topics:')
        self.get_logger().info(f'  Color: {self.color_image_topic}')
        self.get_logger().info(f'  Depth: {self.depth_image_topic}')
        self.get_logger().info(f'  Info: {self.camera_info_topic}')
    
    def color_image_callback(self, msg: Image):
        """Callback for color image"""
        self.latest_color_image = msg
        self._check_images_ready()
    
    def depth_image_callback(self, msg: Image):
        """Callback for depth image"""
        self.latest_depth_image = msg
        self._check_images_ready()
    
    def camera_info_callback(self, msg: CameraInfo):
        """Callback for camera info"""
        self.latest_camera_info = msg
        self._check_images_ready()
    
    
    def _check_images_ready(self):
        """Check if all required images are available"""
        if (self.latest_color_image is not None and 
            self.latest_depth_image is not None and 
            self.latest_camera_info is not None):
            if not self.images_ready:
                self.images_ready = True
                self.get_logger().info('Camera data ready for grasp detection')
    
    def _load_model(self) -> GraspNet:
        """Load GraspNet model"""
        if not self.checkpoint_path:
            self.get_logger().error('Checkpoint path not provided!')
            raise ValueError('checkpoint_path parameter must be set')
            
        # Initialize network
        net = GraspNet(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False
        )
        
        # Load weights
        net.to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        
        self.get_logger().info(f'Model loaded from: {self.checkpoint_path}')
        return net
    
    def get_transform_matrix(self, target_frame: str, source_frame: str) -> Optional[np.ndarray]:
        """Get transformation matrix from source_frame to target_frame"""
        try:
            # Get transform from TF tree
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, 
                Time(), timeout=Duration(seconds=1.0)
            )
            
            # Extract translation
            trans = transform.transform.translation
            translation = np.array([trans.x, trans.y, trans.z])
            
            # Extract rotation (quaternion to rotation matrix)
            rot = transform.transform.rotation
            from scipy.spatial.transform import Rotation as R
            rotation_matrix = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
            
            # Create 4x4 transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = translation
            
            return transform_matrix
            
        except TransformException as ex:
            self.get_logger().warn(f'Could not get transform from {source_frame} to {target_frame}: {ex}')
            return None
    
    def process_rgbd_data(self,
                         color_msg: Image,
                         depth_msg: Image,
                         camera_info: CameraInfo,
                         workspace_min: Optional[Point] = None,
                         workspace_max: Optional[Point] = None,
                         target_object_class: Optional[str] = None) -> Tuple[torch.Tensor, o3d.geometry.PointCloud]:
        """Process RGBD data to generate point cloud"""
        
        # Convert ROS messages to numpy arrays
        color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        
        # Debug image information
        self.get_logger().info(f'Image dimensions:')
        self.get_logger().info(f'  Color image: {color_image.shape}')
        self.get_logger().info(f'  Depth image: {depth_image.shape}')
        self.get_logger().info(f'  Configured depth params: {self.depth_camera_params["width"]}x{self.depth_camera_params["height"]}')
        
        # Check if we're using registered depth (aligned to RGB camera)
        # If depth image has same resolution as RGB, use RGB camera intrinsics
        if depth_image.shape == color_image.shape[:2]:
            self.get_logger().info('Detected registered depth image (aligned to RGB)')
            self.get_logger().info(f'Using RGB camera intrinsics: fx={self.rgb_camera_params["fx"]:.2f}, fy={self.rgb_camera_params["fy"]:.2f}')
            fx = self.rgb_camera_params['fx']
            fy = self.rgb_camera_params['fy']
            cx = self.rgb_camera_params['cx']
            cy = self.rgb_camera_params['cy']
        else:
            self.get_logger().info('Detected native depth image (original resolution)')
            self.get_logger().info(f'Using depth camera intrinsics: fx={self.depth_camera_params["fx"]:.2f}, fy={self.depth_camera_params["fy"]:.2f}')
            fx = self.depth_camera_params['fx']
            fy = self.depth_camera_params['fy']
            cx = self.depth_camera_params['cx']
            cy = self.depth_camera_params['cy']
        
        # Create GraspNet camera object using depth camera parameters
        h, w = depth_image.shape
        # Scale factor for depth values (typically 1000.0 for depth in mm to meters)
        scale = 1000.0
        camera = GraspNetCameraInfo(w, h, fx, fy, cx, cy, scale)
        
        # Generate point cloud
        cloud = create_point_cloud_from_depth_image(depth_image, camera, organized=True)
        
        # Create workspace mask with intelligent segmentation
        import cv2
        import os
        try:
            # Add project root to path for cv_segmentation import
            # project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            # sys.path.append(os.path.join(project_root, 'utils'))

            sys.path.append('/home/roar/graspnet/graspnet-baseline/kinova_graspnet_ros2/utils')

        
            from cv_segmentation import segment_objects

            
            # Determine target object class
            target_class = target_object_class if target_object_class else self.default_target_object_class
            target_class = target_class if target_class else None  # Use None for all objects
            
            # Use intelligent segmentation
            if target_class:
                self.get_logger().info(f'Using intelligent segmentation for object: {target_class}')
            else:
                self.get_logger().info('Using intelligent segmentation for all objects')
                
            segmentation_result = segment_objects(
                color_image,
                target_class=target_class,
                interactive=False,  # No interaction in service mode
                output_path=None,
                return_vis=True  # Get YOLO visualization
            )
            
            # Handle return value based on whether visualization was requested
            if isinstance(segmentation_result, tuple):
                segmentation_mask, yolo_vis_img = segmentation_result
                
                # Publish YOLO detection visualization
                try:
                    yolo_msg = self.bridge.cv2_to_imgmsg(yolo_vis_img, "bgr8")
                    yolo_msg.header.stamp = self.get_clock().now().to_msg()
                    yolo_msg.header.frame_id = self.rgb_camera_frame
                    self.yolo_detection_pub.publish(yolo_msg)
                    self.get_logger().info('Published YOLO detection visualization')
                except Exception as e:
                    self.get_logger().warn(f'Failed to publish YOLO visualization: {e}')
            else:
                segmentation_mask = segmentation_result
            
            # 详细调试信息 - 原始分割结果
            self.get_logger().info(f'=== SEGMENTATION DEBUG ===')
            self.get_logger().info(f'Original segmentation_mask type: {type(segmentation_mask)}')
            if segmentation_mask is not None:
                self.get_logger().info(f'Original segmentation_mask shape: {segmentation_mask.shape}')
                self.get_logger().info(f'Original segmentation_mask dtype: {segmentation_mask.dtype}')
                self.get_logger().info(f'Original segmentation_mask min/max: {np.min(segmentation_mask)}/{np.max(segmentation_mask)}')
                self.get_logger().info(f'Original segmentation_mask unique values: {np.unique(segmentation_mask)}')
                self.get_logger().info(f'Original segmentation_mask sum: {np.sum(segmentation_mask)}')
                self.get_logger().info(f'Original segmentation_mask non-zero pixels: {np.count_nonzero(segmentation_mask)}')
            else:
                self.get_logger().error('segmentation_mask is None!')
            
            # Combine segmentation with geometric constraints
            # Resize segmentation mask to match depth image dimensions
            if segmentation_mask.shape[:2] != depth_image.shape:
                self.get_logger().info(f'Resizing segmentation mask from {segmentation_mask.shape[:2]} to {depth_image.shape}')
                segmentation_mask_resized = cv2.resize(segmentation_mask.astype(np.uint8), 
                                                     (depth_image.shape[1], depth_image.shape[0]), 
                                                     interpolation=cv2.INTER_NEAREST)
                
                # 调试调整大小后的掩码
                self.get_logger().info(f'Resized segmentation_mask shape: {segmentation_mask_resized.shape}')
                self.get_logger().info(f'Resized segmentation_mask sum: {np.sum(segmentation_mask_resized)}')
                self.get_logger().info(f'Resized segmentation_mask unique values: {np.unique(segmentation_mask_resized)}')
                
                workspace_mask = (segmentation_mask_resized > 0).astype(bool)
            else:
                self.get_logger().info('No resizing needed, using original segmentation mask')
                workspace_mask = (segmentation_mask > 0).astype(bool)
            
            # 调试最终工作空间掩码
            self.get_logger().info(f'Final workspace_mask shape: {workspace_mask.shape}')
            self.get_logger().info(f'Final workspace_mask dtype: {workspace_mask.dtype}')
            self.get_logger().info(f'Final workspace_mask sum: {np.sum(workspace_mask)}')
            self.get_logger().info(f'=== END SEGMENTATION DEBUG ===')
            
            # 保存中间结果用于调试
            try:
                debug_dir = "/tmp/graspnet_debug"
                os.makedirs(debug_dir, exist_ok=True)
                
                # 保存调整大小前的掩码（如果有的话）
                if segmentation_mask.shape[:2] != depth_image.shape:
                    original_vis = (segmentation_mask * 255).astype(np.uint8)
                    cv2.imwrite(f"{debug_dir}/segmentation_mask_before_resize.png", original_vis)
                    
                    resized_vis = (segmentation_mask_resized * 255).astype(np.uint8)
                    cv2.imwrite(f"{debug_dir}/segmentation_mask_after_resize.png", resized_vis)
                
            except Exception as debug_save_error:
                self.get_logger().warn(f'Failed to save debug intermediate masks: {debug_save_error}')
            
            # Save debug images for analysis
            try:
                debug_dir = "/tmp/graspnet_debug"
                import os
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save original RGB image
                cv2.imwrite(f"{debug_dir}/rgb_image.png", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
                
                # Save depth image (normalized for visualization)
                depth_vis = (depth_image / np.max(depth_image) * 255).astype(np.uint8)
                cv2.imwrite(f"{debug_dir}/depth_image.png", depth_vis)
                
                # Save original segmentation mask
                seg_vis = (segmentation_mask * 255).astype(np.uint8)
                cv2.imwrite(f"{debug_dir}/segmentation_mask_original.png", seg_vis)
                
                # Save resized segmentation mask
                if segmentation_mask.shape[:2] != depth_image.shape:
                    seg_resized_vis = (segmentation_mask_resized * 255).astype(np.uint8)
                    cv2.imwrite(f"{debug_dir}/segmentation_mask_resized.png", seg_resized_vis)
                
                # Save workspace mask
                workspace_vis = (workspace_mask * 255).astype(np.uint8)
                cv2.imwrite(f"{debug_dir}/workspace_mask.png", workspace_vis)
                
                # Save depth validity mask
                depth_valid = (depth_image > 0).astype(np.uint8) * 255
                cv2.imwrite(f"{debug_dir}/depth_valid_mask.png", depth_valid)
                
                self.get_logger().info(f'Debug images saved to {debug_dir}')
                
            except Exception as save_error:
                self.get_logger().warn(f'Failed to save debug images: {save_error}')
            
            # 检查几何约束参数
            self.get_logger().info(f'Workspace constraints check:')
            self.get_logger().info(f'  workspace_min: {workspace_min}')
            self.get_logger().info(f'  workspace_max: {workspace_max}')
            
            # 暂时跳过几何约束进行测试
            if False:  # 暂时禁用几何约束
                self.get_logger().info(f'Applying geometric constraints...')
                self.get_logger().info(f'  min: ({workspace_min.x}, {workspace_min.y}, {workspace_min.z})')
                self.get_logger().info(f'  max: ({workspace_max.x}, {workspace_max.y}, {workspace_max.z})')
                
                mask_x = (cloud[:, :, 0] >= workspace_min.x) & (cloud[:, :, 0] <= workspace_max.x)
                mask_y = (cloud[:, :, 1] >= workspace_min.y) & (cloud[:, :, 1] <= workspace_max.y)
                mask_z = (cloud[:, :, 2] >= workspace_min.z) & (cloud[:, :, 2] <= workspace_max.z)
                geometric_mask = mask_x & mask_y & mask_z
                
                self.get_logger().info(f'  geometric_mask pixels: {np.sum(geometric_mask)}')
                self.get_logger().info(f'  before geometric constraint: {np.sum(workspace_mask)} pixels')
                
                workspace_mask = workspace_mask & geometric_mask
                
                self.get_logger().info(f'  after geometric constraint: {np.sum(workspace_mask)} pixels')
            else:
                self.get_logger().info('Geometric constraints temporarily disabled for testing')
                
            self.get_logger().info(f'Smart segmentation successful, segmented pixels: {np.sum(workspace_mask)}')
            
        except Exception as seg_error:
            self.get_logger().warn(f'Intelligent segmentation failed, using fallback: {seg_error}')
            
            # Fallback to simple geometric mask
            workspace_mask = np.ones_like(depth_image, dtype=bool)
            if workspace_min and workspace_max:
                mask_x = (cloud[:, :, 0] >= workspace_min.x) & (cloud[:, :, 0] <= workspace_max.x)
                mask_y = (cloud[:, :, 1] >= workspace_min.y) & (cloud[:, :, 1] <= workspace_max.y)
                mask_z = (cloud[:, :, 2] >= workspace_min.z) & (cloud[:, :, 2] <= workspace_max.z)
                workspace_mask = mask_x & mask_y & mask_z
        
        # Process color
        color = color_image.astype(np.float32) / 255.0
        
        # Get valid points
        depth_valid_mask = (depth_image > 0)
        mask = workspace_mask & depth_valid_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        
        # Debug information
        self.get_logger().info(f'Debug overlap analysis:')
        self.get_logger().info(f'  Workspace mask pixels: {np.sum(workspace_mask)}')
        self.get_logger().info(f'  Depth valid pixels: {np.sum(depth_valid_mask)}')
        self.get_logger().info(f'  Overlap pixels: {np.sum(mask)}')
        self.get_logger().info(f'  Workspace mask shape: {workspace_mask.shape}')
        self.get_logger().info(f'  Depth image shape: {depth_image.shape}')
        
        # Sample point cloud
        if len(cloud_masked) == 0:
            raise ValueError(f"No valid points found in segmented region. Segmented pixels: {np.sum(workspace_mask)}, Valid depth pixels: {np.sum(depth_image > 0)}, Overlap: {np.sum(mask)}")
        elif len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        
        cloud_sampled = cloud_masked[idxs]
        
        # Convert to torch tensor
        cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        cloud_tensor = cloud_tensor.to(self.device)
        
        # Create Open3D point cloud
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        o3d_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        
        return cloud_tensor, o3d_cloud
    
    def predict_grasps(self, point_cloud: torch.Tensor) -> GraspGroup:
        """Predict grasp poses"""
        with torch.no_grad():
            end_points = {'point_clouds': point_cloud}
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        
        gg_array = grasp_preds[0].detach().cpu().numpy()
        return GraspGroup(gg_array)
    
    def filter_collisions(self, grasps: GraspGroup, cloud: o3d.geometry.PointCloud) -> GraspGroup:
        """Filter grasps by collision detection"""
        if self.collision_thresh <= 0:
            return grasps
            
        mfcdetector = ModelFreeCollisionDetector(
            np.array(cloud.points),
            voxel_size=self.voxel_size
        )
        collision_mask = mfcdetector.detect(
            grasps,
            approach_dist=0.05,
            collision_thresh=self.collision_thresh
        )
        
        return grasps[~collision_mask]
    
    
    def grasp_to_pose_stamped(self, grasp_array: np.ndarray, frame_id: str) -> PoseStamped:
        """Convert grasp array to PoseStamped message"""
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = frame_id
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Extract translation
        pose_msg.pose.position.x = float(grasp_array[13])
        pose_msg.pose.position.y = float(grasp_array[14])
        pose_msg.pose.position.z = float(grasp_array[15])
        
        # Extract rotation matrix and convert to quaternion
        rotation_matrix = grasp_array[4:13].reshape(3, 3)
        
        # Convert rotation matrix to quaternion using scipy
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  # [x, y, z, w]
        
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        return pose_msg
    
    def _should_use_internal_images(self, request: DetectGrasps.Request) -> bool:
        """Determine whether to use internal subscribed images or request images"""
        # Check if request has empty/default images (indicating client wants to use internal data)
        request_has_images = (
            hasattr(request.color_image, 'data') and len(request.color_image.data) > 0 and
            hasattr(request.depth_image, 'data') and len(request.depth_image.data) > 0
        )
        
        # Use internal images if:
        # 1. Images are ready from subscription AND
        # 2. Either request has no image data OR we prefer internal data
        if self.images_ready and not request_has_images:
            return True
        elif self.images_ready and request_has_images:
            # Both available, use internal (more recent) data
            return True
        elif not self.images_ready and request_has_images:
            # Only request data available
            return False
    
    def visualize_grasps_on_image(self, color_image: np.ndarray, grasp_group: GraspGroup, 
                                 depth_image: np.ndarray, camera_intrinsics: dict) -> np.ndarray:
        """
        在RGB图像上可视化抓取姿态
        
        Args:
            color_image: RGB图像
            grasp_group: GraspNet推理得到的抓取组
            depth_image: 深度图像
            camera_intrinsics: 相机内参
            
        Returns:
            带有抓取可视化的图像
        """
        import cv2
        vis_image = color_image.copy()
        
        if len(grasp_group) == 0:
            return vis_image
            
        # 获取前几个最好的抓取
        top_grasps = grasp_group.nms().sort_by_score()[:5]
        
        fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
        cx, cy = camera_intrinsics['cx'], camera_intrinsics['cy']
        
        colors = [
            (0, 255, 0),    # 绿色 - 最佳
            (0, 255, 255),  # 黄色
            (255, 255, 0),  # 青色  
            (255, 0, 255),  # 品红
            (255, 0, 0),    # 红色
        ]
        
        for i, (translation, rotation_matrix, width, score) in enumerate(
            zip(top_grasps.translations, top_grasps.rotation_matrices, 
                top_grasps.widths, top_grasps.scores)):
            
            if i >= len(colors):
                break
                
            color = colors[i]
            
            # 将3D抓取点投影到图像上
            if translation[2] > 0:  # 确保在相机前方
                u = int(fx * translation[0] / translation[2] + cx)
                v = int(fy * translation[1] / translation[2] + cy)
                
                if 0 <= u < vis_image.shape[1] and 0 <= v < vis_image.shape[0]:
                    # 绘制抓取中心点
                    cv2.circle(vis_image, (u, v), 8, color, -1)
                    
                    # 绘制抓取方向 (从旋转矩阵提取方向)
                    approach_vector = rotation_matrix[:, 2]  # Z轴为接近方向
                    gripper_direction = rotation_matrix[:, 1]  # Y轴为夹爪张开方向
                    
                    # 投影抓取接近方向
                    approach_3d = translation + approach_vector * 0.05
                    if approach_3d[2] > 0:
                        u_approach = int(fx * approach_3d[0] / approach_3d[2] + cx)
                        v_approach = int(fy * approach_3d[1] / approach_3d[2] + cy)
                        cv2.arrowedLine(vis_image, (u, v), (u_approach, v_approach), color, 3)
                    
                    # 绘制夹爪张开方向
                    width_scale = width * 0.5
                    left_3d = translation + gripper_direction * width_scale
                    right_3d = translation - gripper_direction * width_scale
                    
                    if left_3d[2] > 0 and right_3d[2] > 0:
                        u_left = int(fx * left_3d[0] / left_3d[2] + cx)
                        v_left = int(fy * left_3d[1] / left_3d[2] + cy)
                        u_right = int(fx * right_3d[0] / right_3d[2] + cx)
                        v_right = int(fy * right_3d[1] / right_3d[2] + cy)
                        
                        cv2.line(vis_image, (u_left, v_left), (u_right, v_right), color, 3)
                    
                    # 添加分数文本
                    cv2.putText(vis_image, f'{score:.2f}', 
                              (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, color, 2)
        
        return vis_image
    
    def publish_grasp_markers(self, grasp_group: GraspGroup, frame_id: str = 'camera_depth_frame'):
        """
        发布RViz中的抓取可视化标记
        
        Args:
            grasp_group: GraspNet推理得到的抓取组  
            frame_id: 坐标系ID
        """
        marker_array = MarkerArray()
        
        # 清除之前的标记
        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = ""
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        if len(grasp_group) == 0:
            self.grasp_marker_pub.publish(marker_array)
            return
            
        # 获取前5个最好的抓取
        top_grasps = grasp_group.nms().sort_by_score()[:5]
        
        for i, (translation, rotation_matrix, width, score) in enumerate(
            zip(top_grasps.translations, top_grasps.rotation_matrices, 
                top_grasps.widths, top_grasps.scores)):
            
            # 创建抓取姿态
            pose = Pose()
            pose.position = Point(x=float(translation[0]), 
                                y=float(translation[1]), 
                                z=float(translation[2]))
            
            # 将旋转矩阵转换为四元数
            from scipy.spatial.transform import Rotation as R
            rotation = R.from_matrix(rotation_matrix)
            quat = rotation.as_quat()  # [x, y, z, w]
            pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]), 
                                        z=float(quat[2]), w=float(quat[3]))
            
            # 创建夹爪标记
            markers = self.create_gripper_marker(pose, width, score, i * 5, frame_id)
            marker_array.markers.extend(markers)
        
        self.grasp_marker_pub.publish(marker_array)
        self.get_logger().info(f'Published {len(top_grasps)} grasp markers')
    
    def create_gripper_marker(self, pose: Pose, width: float, score: float, 
                             marker_id: int, frame_id: str) -> list:
        """创建夹爪可视化标记"""
        markers = []
        
        # 根据分数设置颜色 (红到绿)
        from std_msgs.msg import ColorRGBA
        from geometry_msgs.msg import Vector3
        
        color = ColorRGBA()
        color.r = 1.0 - score
        color.g = score  
        color.b = 0.0
        color.a = 0.8
        
        # 夹爪手掌
        palm_marker = Marker()
        palm_marker.header.frame_id = frame_id
        palm_marker.header.stamp = self.get_clock().now().to_msg()
        palm_marker.ns = "gripper_palm"
        palm_marker.id = marker_id
        palm_marker.type = Marker.CUBE
        palm_marker.action = Marker.ADD
        palm_marker.pose = pose
        palm_marker.scale = Vector3(x=0.02, y=0.08, z=0.04)
        palm_marker.color = color
        palm_marker.lifetime.sec = 30  # 设置标记存活时间30秒
        markers.append(palm_marker)
        
        # 夹爪手指
        from scipy.spatial.transform import Rotation as R
        q = pose.orientation
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        rotation_matrix = rotation.as_matrix()
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        
        # 左手指
        left_finger = Marker()
        left_finger.header = palm_marker.header
        left_finger.ns = "gripper_left_finger"
        left_finger.id = marker_id + 1
        left_finger.type = Marker.CUBE
        left_finger.action = Marker.ADD
        
        left_offset = rotation_matrix @ np.array([0, width/2, 0.04])
        left_pos = position + left_offset
        left_finger.pose.position = Point(x=left_pos[0], y=left_pos[1], z=left_pos[2])
        left_finger.pose.orientation = pose.orientation
        left_finger.scale = Vector3(x=0.01, y=0.01, z=0.08)
        left_finger.color = color
        left_finger.lifetime.sec = 30
        markers.append(left_finger)
        
        # 右手指
        right_finger = Marker()
        right_finger.header = palm_marker.header
        right_finger.ns = "gripper_right_finger"
        right_finger.id = marker_id + 2
        right_finger.type = Marker.CUBE
        right_finger.action = Marker.ADD
        
        right_offset = rotation_matrix @ np.array([0, -width/2, 0.04])
        right_pos = position + right_offset
        right_finger.pose.position = Point(x=right_pos[0], y=right_pos[1], z=right_pos[2])
        right_finger.pose.orientation = pose.orientation
        right_finger.scale = Vector3(x=0.01, y=0.01, z=0.08)
        right_finger.color = color
        right_finger.lifetime.sec = 30
        markers.append(right_finger)
        
        # 接近方向箭头
        approach_marker = Marker()
        approach_marker.header = palm_marker.header
        approach_marker.ns = "approach_vector"
        approach_marker.id = marker_id + 3
        approach_marker.type = Marker.ARROW
        approach_marker.action = Marker.ADD
        
        approach_vector = -rotation_matrix[:, 2]  # 负Z轴
        approach_start = position + approach_vector * 0.1
        approach_marker.points = [
            Point(x=approach_start[0], y=approach_start[1], z=approach_start[2]),
            Point(x=position[0], y=position[1], z=position[2])
        ]
        approach_marker.scale = Vector3(x=0.005, y=0.01, z=0.01)
        approach_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
        approach_marker.lifetime.sec = 30
        markers.append(approach_marker)
        
        # 分数文本
        score_marker = Marker()
        score_marker.header = palm_marker.header
        score_marker.ns = "grasp_score"
        score_marker.id = marker_id + 4
        score_marker.type = Marker.TEXT_VIEW_FACING
        score_marker.action = Marker.ADD
        
        text_pos = position + np.array([0, 0, -0.05])
        score_marker.pose.position = Point(x=text_pos[0], y=text_pos[1], z=text_pos[2])
        score_marker.scale.z = 0.02
        score_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        score_marker.text = f"{score:.2f}"
        score_marker.lifetime.sec = 30
        markers.append(score_marker)
        
        return markers
    
    def detect_grasps_callback(self, request: DetectGrasps.Request, response: DetectGrasps.Response):
        """Service callback for grasp detection"""
        self.get_logger().info('DEBUG: === GRASP DETECTION CALLBACK STARTED ===')
        
        try:
            start_time = time.time()
            
            # Process RGBD data
            # Determine image source
            use_internal_images = self._should_use_internal_images(request)
            
            if use_internal_images:
                self.get_logger().info('Using internally subscribed camera data')
                color_image = self.latest_color_image
                depth_image = self.latest_depth_image
                camera_info = self.latest_camera_info
            else:
                self.get_logger().info('Using images from service request')
                color_image = request.color_image
                depth_image = request.depth_image
                camera_info = request.camera_info
            
            # Process RGBD data
            point_cloud, o3d_cloud = self.process_rgbd_data(
                    color_image,
                    depth_image,
                    camera_info,
                    request.workspace_min if request.workspace_min else None,
                    request.workspace_max if request.workspace_max else None,
                    request.target_object_class if request.target_object_class else None
                )
            
            # Predict grasps
            grasps = self.predict_grasps(point_cloud)
            response.total_grasps_detected = len(grasps)
            
            # Filter collisions
            filtered_grasps = self.filter_collisions(grasps, o3d_cloud)
            response.grasps_after_filtering = len(filtered_grasps)
            
            # NMS and sorting - Re-enabled to reduce position variation
            filtered_grasps.nms()  # NMS filters out similar/overlapping grasps
            filtered_grasps.sort_by_score()
            
            # Get top grasps
            max_grasps = request.max_grasps if request.max_grasps > 0 else 10
            top_grasps = filtered_grasps[:min(max_grasps, len(filtered_grasps))]
            
            # Convert to ROS messages using GraspGroup properties
            # 根据GraspNet API文档，使用单独的属性而不是grasp_array
            translations = top_grasps.translations
            rotation_matrices = top_grasps.rotation_matrices  
            scores = top_grasps.scores
            widths = top_grasps.widths
            depths = top_grasps.depths
            
            self.get_logger().info(f'Processing {len(translations)} grasps for ROS conversion')
            
            # ===================== GraspNet坐标系转换 =====================
            # GraspNet坐标系: X轴=接近方向, Y轴=夹爪开合, Z轴=向上
            # TF相机坐标系: X轴=右, Y轴=下, Z轴=前(接近方向)
            # 需要将GraspNet的X轴(接近)转换为TF的Z轴(接近)
            # 将GraspNet的Y轴(夹爪开合)转换为TF的X轴(夹爪开合)
            from scipy.spatial.transform import Rotation as R

            graspnet_to_tf_rotation = np.array([
                [1,  0,  0],   
                [0,  1,  0],   
                [0,  0,  1]    
            ])

            # 构建4x4齐次变换矩阵（坐标系变换，只有旋转，无平移）
            graspnet_to_tf_transform = np.eye(4)
            graspnet_to_tf_transform[:3, :3] = graspnet_to_tf_rotation
            graspnet_to_tf_transform[:3, 3] = [0, 0, 0]  # 原点重合，无平移
            
            self.get_logger().info('Applying GraspNet to TF coordinate system transformation using homogeneous matrices')
            
            for i in range(len(translations)):
                # ===== 齐次变换矩阵坐标系转换 =====
                # 构建GraspNet坐标系中的抓取齐次变换矩阵
                grasp_transform_graspnet = np.eye(4)
                grasp_transform_graspnet[:3, :3] = rotation_matrices[i]  # 旋转部分
                grasp_transform_graspnet[:3, 3] = [translations[i][0], translations[i][1], translations[i][2]]  # 平移部分
                
                # 坐标系变换：左乘变换矩阵
                grasp_transform_tf = graspnet_to_tf_transform @ grasp_transform_graspnet
                
                # 提取转换后的位置和姿态
                pos_tf_camera = grasp_transform_tf[:3, 3]
                rot_matrix_tf_camera = grasp_transform_tf[:3, :3]
                
                # 构建ROS消息
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = self.depth_camera_frame
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                
                # 设置转换后的位置
                pose_msg.pose.position.x = float(pos_tf_camera[0])
                pose_msg.pose.position.y = float(pos_tf_camera[1]) 
                pose_msg.pose.position.z = float(pos_tf_camera[2])
                
                # 设置转换后的方向（旋转矩阵转四元数）
                rotation = R.from_matrix(rot_matrix_tf_camera)
                quat = rotation.as_quat()  # [x, y, z, w]
                
                pose_msg.pose.orientation.x = quat[0]
                pose_msg.pose.orientation.y = quat[1]
                pose_msg.pose.orientation.z = quat[2]
                pose_msg.pose.orientation.w = quat[3]
                
                response.grasp_poses.append(pose_msg)
                
                # Grasp properties
                response.grasp_scores.append(float(scores[i]))
                response.grasp_widths.append(float(widths[i]))
                response.grasp_depths.append(float(depths[i]))
            
            response.inference_time = time.time() - start_time
            response.success = len(response.grasp_poses) > 0
            
            self.get_logger().info(f'Debug: About to publish visualization. response.success = {response.success}')
            
            # 构建简化的响应消息
            if len(response.grasp_poses) > 0:
                message_lines = []
                message_lines.append("\n" + "="*80)
                message_lines.append("🎯 抓取检测结果 - 前2名最佳抓取")
                message_lines.append("="*80)
                
                top_n = min(2, len(response.grasp_poses))
                for i in range(top_n):
                    pose = response.grasp_poses[i]
                    score = response.grasp_scores[i]
                    width = response.grasp_widths[i]
                    
                    message_lines.append(f"\n🥇 抓取 #{i+1} (得分: {score:.3f})")
                    message_lines.append(f"位置: ({pose.pose.position.x:.6f}, {pose.pose.position.y:.6f}, {pose.pose.position.z:.6f})")
                    message_lines.append(f"姿态: ({pose.pose.orientation.x:.6f}, {pose.pose.orientation.y:.6f}, {pose.pose.orientation.z:.6f}, {pose.pose.orientation.w:.6f})")
                    message_lines.append(f"夹爪宽度: {width:.6f}m")
                    
                    # 生成可直接执行的命令
                    command = f'''ros2 service call /execute_grasp kinova_graspnet_ros2/srv/ExecuteGrasp "{{
  grasp_pose: {{
    header: {{frame_id: '{pose.header.frame_id}'}},
    pose: {{
      position: {{x: {pose.pose.position.x}, y: {pose.pose.position.y}, z: {pose.pose.position.z}}},
      orientation: {{x: {pose.pose.orientation.x}, y: {pose.pose.orientation.y}, z: {pose.pose.orientation.z}, w: {pose.pose.orientation.w}}}
    }}
  }},
  grasp_width: {width},
  approach_distance: 0.1,
  max_velocity_scaling: 0.3,
  max_acceleration_scaling: 0.3
}}"'''
                    
                    message_lines.append(f"\n📋 执行命令 #{i+1}:")
                    message_lines.append(command)
                
                message_lines.append("\n" + "="*80)
                message_lines.append(f"✅ 检测完成: 共{len(response.grasp_poses)}个抓取，耗时{response.inference_time:.3f}s")
                message_lines.append("="*80)
                
                response.message = "\n".join(message_lines)
            else:
                response.message = "\n❌ 未检测到有效抓取"
            
            # 调试信息：检查抓取数量
            self.get_logger().info(f'Debug: filtered_grasps length = {len(filtered_grasps)}')
            self.get_logger().info(f'Debug: response.grasp_poses length = {len(response.grasp_poses)}')
            
            # 发布RViz标记（始终尝试发布，不依赖图像数据）
            if len(filtered_grasps) > 0:
                try:
                    self.get_logger().info(f'Attempting to publish {len(filtered_grasps)} grasp markers')
                    self.publish_grasp_markers(filtered_grasps, self.depth_camera_frame)
                    self.get_logger().info('Successfully published grasp markers')
                except Exception as e:
                    self.get_logger().error(f'Failed to publish grasp markers: {e}')
                    import traceback
                    self.get_logger().error(f'Traceback: {traceback.format_exc()}')
            else:
                self.get_logger().warn('No filtered grasps to publish')
            
            # 发布图像可视化（仅在有图像数据时）
            if len(filtered_grasps) > 0 and color_image is not None and depth_image is not None:
                try:
                    # 先将ROS图像消息转换为numpy数组
                    if hasattr(color_image, 'data'):  # 是ROS消息
                        color_np = self.bridge.imgmsg_to_cv2(color_image, desired_encoding='rgb8')
                        depth_np = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
                    else:  # 已经是numpy数组
                        color_np = color_image
                        depth_np = depth_image
                        
                    # 在RGB图像上绘制抓取 - 根据深度图像是否对齐选择内参
                    # 如果深度图像已对齐到RGB（尺寸相同），使用RGB内参；否则使用深度内参
                    camera_intrinsics = (self.rgb_camera_params if depth_np.shape == color_np.shape[:2] 
                                        else self.depth_camera_params)
                    vis_image = self.visualize_grasps_on_image(
                        color_np, filtered_grasps, depth_np, camera_intrinsics
                    )
                    
                    # 保存可视化图像用于调试
                    import cv2
                    cv2.imwrite('/tmp/graspnet_debug/grasp_visualization.png', vis_image)
                    self.get_logger().info('Saved grasp visualization to /tmp/graspnet_debug/grasp_visualization.png')
                    
                    # 确保图像是BGR格式（OpenCV默认）
                    if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
                        # 从RGB转换为BGR用于cv2显示
                        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                    else:
                        vis_image_bgr = vis_image
                    vis_msg = self.bridge.cv2_to_imgmsg(vis_image_bgr, 'bgr8')
                    vis_msg.header.stamp = self.get_clock().now().to_msg()
                    vis_msg.header.frame_id = self.rgb_camera_frame
                    self.debug_image_pub.publish(vis_msg)
                    self.get_logger().info('Published visualization image')
                except Exception as e:
                    self.get_logger().warn(f'Failed to publish visualization image: {e}')
            
            # 简化的节点日志输出
            self.get_logger().info(f'Grasp detection completed: {len(response.grasp_poses)} grasps in {response.inference_time:.3f}s')
            
        except Exception as e:
            self.get_logger().error(f'Grasp detection failed: {str(e)}')
            response.success = False
            response.message = f"Detection failed: {str(e)}"
        
        return response
    


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = GraspDetectionService()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()