"""
智能分割模块 - 集成YOLO-World和SAM
基于Dehao-Zhou项目的cv_process.py核心功能迁移

核心功能:
1. YOLO-World进行物体检测
2. SAM进行精确分割
3. 支持指定类别或手动选择
4. 生成分割掩码替代预标注数据

使用方法:
    from utils.cv_segmentation import segment_objects
    mask = segment_objects(image, target_class="bottle")
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor


class SmartSegmentation:
    """智能分割类 - 集成YOLO和SAM"""
    
    def __init__(self, sam_model='sam_b.pt', yolo_model='yolov8s-world.pt', device=None):
        """
        初始化分割模型
        
        Args:
            sam_model: SAM模型权重路径
            yolo_model: YOLO-World模型路径
        """
        self.sam_model_path = sam_model
        self.yolo_model_path = yolo_model
        self._sam_predictor = None
        self._yolo_model = None

        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
    def _init_sam(self):
        """延迟初始化SAM模型"""
        if self._sam_predictor is None:
            overrides = dict(
                task='segment',
                mode='predict',
                model=self.sam_model_path,
                conf=0.01,
                save=False
            )
            self._sam_predictor = SAMPredictor(overrides=overrides)
        return self._sam_predictor
    
    def _init_yolo(self):
        """延迟初始化YOLO模型"""
        if self._yolo_model is None:
            self._yolo_model = YOLO(self.yolo_model_path).to(self.device)
        return self._yolo_model
    
    def detect_objects(self, image, target_class=None, conf_threshold=0.25):
        """
        使用YOLO-World检测物体
        
        Args:
            image: 输入图像 (numpy array或文件路径)
            target_class: 目标类别名称 (可选)
            conf_threshold: 置信度阈值
            
        Returns:
            list: 检测结果列表
            numpy.ndarray: 可视化图像
        """
        model = self._init_yolo()
        
        # 设置检测类别
        if target_class:
            model.set_classes([target_class])
        
        # 执行检测
        results = model.predict(image)
        boxes = results[0].boxes
        vis_img = results[0].plot()
        
        # 提取有效检测结果
        valid_detections = []
        if boxes is not None:
            for box in boxes:
                if box.conf.item() > conf_threshold:
                    valid_detections.append({
                        "xyxy": box.xyxy[0].tolist(),
                        "conf": box.conf.item(),
                        "cls": results[0].names[box.cls.item()]
                    })
        
        return valid_detections, vis_img
    
    def segment_with_sam(self, image, bbox=None, point=None):
        """
        使用SAM进行分割
        
        Args:
            image: RGB图像 (numpy array)
            bbox: 边界框 [x1, y1, x2, y2] (可选)
            point: 点击点 [x, y] (可选)
            
        Returns:
            tuple: (中心点, 分割掩码)
        """
        predictor = self._init_sam()
        
        # 确保图像是RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 假设输入是BGR，转换为RGB
            if isinstance(image, np.ndarray):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = image
        
        # 设置图像
        predictor.set_image(image_rgb)
        
        # 根据输入类型执行分割
        if bbox is not None:
            results = predictor(bboxes=[bbox])
        elif point is not None:
            results = predictor(points=[point], labels=[1])
        else:
            raise ValueError("必须提供bbox或point参数")
        
        return self._process_sam_results(results)
    
    def _process_sam_results(self, results):
        """处理SAM结果"""
        if not results or not results[0].masks:
            return None, None
        
        # 获取第一个掩码
        mask = results[0].masks.data[0].cpu().numpy()
        mask = (mask > 0).astype(np.uint8) * 255
        
        # 计算质心
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask
        
        M = cv2.moments(contours[0])
        if M["m00"] == 0:
            return None, mask
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return (cx, cy), mask
    
    def segment_objects(self, image, target_class=None, interactive=False, output_path=None, return_vis=False):
        """
        完整的物体分割流程
        
        Args:
            image: 输入图像 (numpy array或文件路径)
            target_class: 目标类别 (可选)
            interactive: 是否启用交互模式
            output_path: 输出掩码保存路径
            
        Returns:
            numpy.ndarray: 分割掩码
        """
        # 加载图像
        if isinstance(image, str):
            bgr_img = cv2.imread(image)
            if bgr_img is None:
                raise ValueError(f"无法读取图像: {image}")
            image_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bgr_img = image
        
        # 1. 物体检测
        detections, vis_img = self.detect_objects(bgr_img, target_class)
        
        # 保存检测可视化
        if output_path:
            detection_path = output_path.replace('.png', '_detection.jpg')
            cv2.imwrite(detection_path, vis_img)
        
        # 2. 分割策略选择
        if detections:
            # 自动选择最高置信度的检测结果
            best_detection = max(detections, key=lambda x: x["conf"])
            center, mask = self.segment_with_sam(image_rgb, bbox=best_detection["xyxy"])
            print(f"自动选择 {best_detection['cls']}, 置信度: {best_detection['conf']:.2f}")
            
        elif interactive:
            # 交互模式：用户点击选择
            print("未检测到目标，请点击选择物体")
            cv2.imshow('点击选择物体', vis_img)
            
            click_point = []
            
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    click_point.extend([x, y])
                    cv2.destroyAllWindows()
            
            cv2.setMouseCallback('点击选择物体', mouse_callback)
            cv2.waitKey(0)
            
            if len(click_point) == 2:
                center, mask = self.segment_with_sam(image_rgb, point=click_point)
            else:
                raise ValueError("未选择任何点")
                
        else:
            # 非交互模式且无检测结果，返回空掩码
            print("警告: 未检测到物体且未启用交互模式")
            h, w = image_rgb.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            if return_vis:
                return mask, vis_img
            return mask
        
        # 3. 保存结果
        if mask is not None and output_path:
            cv2.imwrite(output_path, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
            print(f"分割掩码已保存至: {output_path}")
        
        if return_vis:
            return mask, vis_img
        return mask

    def segment_all_objects(self, image, conf_threshold=0.25):
        """
        分割图像中的所有可识别对象，并按类别返回它们的掩码。

        Args:
            image (np.ndarray): 输入的BGR图像。
            conf_threshold (float): YOLO检测的置信度阈值。

        Returns:
            dict: 一个字典，键是对象类别名(str)，值是该类别下所有掩码(np.ndarray of uint8)的列表。
                  例如: {'bottle': [mask1, mask2], 'cup': [mask3]}
        """
        yolo = self._init_yolo()
        sam = self._init_sam()

        # 1. 使用YOLO-World进行通用物体检测
        # 不设置特定类别，让YOLO-World检测所有它能识别的物体
        yolo.set_classes(["person", "bottle", "cup", "chair", "laptop", "keyboard", "mouse", "book", "cell phone"])
        results = yolo.predict(image, conf=conf_threshold, verbose=False)

        # 如果没有检测结果，返回空字典
        if not results or results[0].boxes is None:
            return {}
            
        # 提取所有检测框
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_indices = results[0].boxes.cls.cpu().numpy()
        class_names_map = results[0].names
        
        # 2. 使用SAM对所有检测框进行分割
        # SAM可以一次性处理多个边界框，效率很高
        sam.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        sam_results = sam(bboxes=boxes)
        
        segmented_objects = {}
        if not sam_results or not sam_results[0].masks:
            return {}

        # 3. 整理分割结果
        all_masks = sam_results[0].masks.data.cpu().numpy()

        for i in range(len(all_masks)):
            class_name = class_names_map[int(class_indices[i])]
            mask = (all_masks[i] > 0).astype(np.uint8)

            if class_name not in segmented_objects:
                segmented_objects[class_name] = []
            
            segmented_objects[class_name].append(mask)

        return segmented_objects

# 全局分割器实例
_global_segmenter = None

# def get_segmenter(device=None):
#     """获取全局分割器实例"""
#     global _global_segmenter
#     if _global_segmenter is None:
#         _global_segmenter = SmartSegmentation(device=device)
#     return _global_segmenter
def get_segmenter(device=None):
    """
    获取一个全新的分割器实例。
    已禁用单例模式以避免YOLO模型的状态污染问题。
    """
    # 每次都创建一个全新的对象，这样内部的YOLO模型也是全新的。
    return SmartSegmentation(device=device)


def segment_objects(image, target_class=None, interactive=True, output_path='mask.png', return_vis=False, device=None):
    """
    便捷函数：物体分割
    
    Args:
        image: 输入图像 (numpy array或文件路径)
        target_class: 目标类别名称
        interactive: 是否启用交互模式
        output_path: 输出掩码路径
        return_vis: 是否返回YOLO检测可视化图像
        
    Returns:
        numpy.ndarray: 分割掩码
        numpy.ndarray: YOLO检测可视化图像 (如果return_vis=True)
    """
    segmenter = get_segmenter(device=device)
    return segmenter.segment_objects(image, target_class, interactive, output_path, return_vis)


# 兼容原始接口
def segment_image(image, output_mask='mask.png'):
    """
    兼容Dehao-Zhou原始接口的包装函数
    
    Args:
        image: 输入图像 (numpy array或文件路径)
        output_mask: 输出掩码文件名
        
    Returns:
        numpy.ndarray: 分割掩码
    """
    return segment_objects(image, interactive=True, output_path=output_mask)


if __name__ == '__main__':
    # 测试代码
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test-all':
        image_path = sys.argv[2]
        print(f"--- 测试全局分割功能 ---")
        print(f"图像: {image_path}")
        
        segmenter = SmartSegmentation()
        bgr_image = cv2.imread(image_path)
        
        all_objects = segmenter.segment_all_objects(bgr_image)
        
        print(f"\n检测并分割到的所有对象类别:")
        total_masks = 0
        for class_name, masks in all_objects.items():
            print(f"  - 类别: {class_name}, 数量: {len(masks)}")
            total_masks += len(masks)
            # 可选：将每个掩码保存下来进行可视化检查
            for i, mask in enumerate(masks):
                cv2.imwrite(f'test_output_{class_name}_{i}.png', mask * 255)

        print(f"\n总共分割出 {total_masks} 个物体。")
    
    # elif len(sys.argv) > 1:
    #     image_path = sys.argv[1]
    #     target_class = sys.argv[2] if len(sys.argv) > 2 else None
        
    #     print(f"分割图像: {image_path}")
    #     if target_class:
    #         print(f"目标类别: {target_class}")
        
    #     mask = segment_objects(image_path, target_class=target_class)
    #     print(f"分割完成，掩码尺寸: {mask.shape}")
    else:
        print("用法: python cv_segmentation.py <image_path> [target_class]")