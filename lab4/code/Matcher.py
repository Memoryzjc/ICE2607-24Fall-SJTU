import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class ImageMatcher:
    def __init__(self, matching_method='bf'):
        """
        初始化图像匹配器
        matching_method: 'bf' 或 'flann'
        """
        self.matching_method = matching_method
        
        # 初始化特征检测器 (使用SIFT)
        self.feature_detector = cv2.SIFT_create()
        
        # 初始化特征匹配器
        if matching_method == 'bf':
            self.matcher = cv2.BFMatcher()
        else:
            # FLANN参数
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            
        self.image_paths = []
        self.image_features = []
        self.image_descriptors = []
        
    def index_images(self, image_dir):
        """
        为图像目录建立索引
        INPUT: 
            image_dir: 图像目录路径
        """
        image_dir = Path(image_dir)
        
        for img_path in image_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg']:
                try:
                    # 读取图像
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # 检测关键点和描述符
                    keypoints, descriptors = self.feature_detector.detectAndCompute(img, None)
                    
                    if descriptors is not None:
                        self.image_paths.append(str(img_path))
                        self.image_features.append(keypoints)
                        self.image_descriptors.append(descriptors)
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    def find_matches(self, query_img_path, k=1):
        """
        查找与查询图像最匹配的k张图像
        INPUT:
            query_img_path: 查询图像路径
            k: 返回的匹配图像数量
        OUTPUT:
            匹配图像的路径列表
        """
        # 读取查询图像
        query_img = cv2.imread(query_img_path)
        if query_img is None:
            raise ValueError("Cannot read query image")
            
        # 检测查询图像的特征
        query_keypoints, query_descriptors = self.feature_detector.detectAndCompute(query_img, None)
        
        if query_descriptors is None:
            return []
            
        # 计算所有图像的匹配分数
        match_scores = []
        
        for idx, db_descriptors in enumerate(self.image_descriptors):
            if self.matching_method == 'bf':
                matches = self.matcher.knnMatch(query_descriptors, db_descriptors, k=2)
            else:
                matches = self.matcher.knnMatch(query_descriptors, db_descriptors, k=2)
                
            # 应用比率测试
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                    
            match_scores.append((len(good_matches), idx))
            
        # 按匹配数量排序
        match_scores.sort(reverse=True)
        
        # 返回最佳匹配的图像路径
        return [self.image_paths[idx] for _, idx in match_scores[:k]]
    
    def visualize_results(self, query_img_path, result_paths, save_path=None):
        """
        可视化匹配结果 \\
        INPUT:
            query_img_path: 查询图像路径
            result_paths: 匹配图像路径列表
            save_path: 可选，保存结果图像的路径
        """
        # 读取并调整查询图像大小
        query_img = cv2.imread(query_img_path)
        
        # 读取并调整结果图像大小
        result_imgs = []
        for path in result_paths:
            img = cv2.imread(path)
            result_imgs.append(img)
            
        # 可视化
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        plt.title("Query Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_imgs[0], cv2.COLOR_BGR2RGB))
        plt.title("Best Match for NN")
        plt.axis('off')

        plt.suptitle("Results for NN")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()