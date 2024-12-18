import cv2
import numpy as np
from collections import defaultdict
import os
from pathlib import Path

class ImageLSH:
    def __init__(self, projection_sets):
        """
        初始化LSH
        projection_sets: List[ List[ int]], 投影集合的列表
        """
        self.projection_sets = projection_sets
        self.hash_tables = [defaultdict(list) for _ in range(len(projection_sets))]
        
    def compute_color_histogram(self, image):
        """
        计算12维颜色直方图特征
        将图像分为4个区域，每个区域计算3维颜色直方图
        """
        # 将图像调整为相同大小以确保一致性
        image = cv2.resize(image, (200, 200))
        
        # 将图像分为2x2的四个区域
        h, w = image.shape[:2]
        mid_h, mid_w = h//2, w//2
        regions = [
            image[0:mid_h, 0:mid_w],      # 左上
            image[0:mid_h, mid_w:w],      # 右上
            image[mid_h:h, 0:mid_w],      # 左下
            image[mid_h:h, mid_w:w]       # 右下
        ]
        
        # 对每个区域计算3维颜色直方图
        hist_features = []
        for region in regions:
            # 计算颜色直方图
            b, g, r = cv2.split(region)
            b_energy = np.sum(b)
            g_energy = np.sum(g)
            r_energy = np.sum(r)
            total_energy = b_energy + g_energy + r_energy
            hist = [b_energy/total_energy, g_energy/total_energy, r_energy/total_energy]
            hist_features.extend(hist)
            
        return np.array(hist_features)
    
    def _quantize_feature(self, feature):
        """
        将特征向量量化为0,1,2三个值
        """
        quantized = np.zeros_like(feature)
        quantized[feature > 0.6] = 2
        quantized[(feature >= 0.3) & (feature <= 0.6)] = 1
        return quantized
    
    def _hash_vector(self, feature, projection):
        """
        计算特征向量在某个投影集合上的哈希值
        """
        quantized = self._quantize_feature(feature)
        hamming = []
        
        # 计算Hamming码
        for i in range(len(quantized)):
            match quantized[i]:
                case 0:
                    hamming.extend([0, 0])
                case 1:
                    hamming.extend([1, 0])
                case 2:
                    hamming.extend([1, 1])

        hamming = np.array(hamming)
        projected = hamming[projection]
        return tuple(projected)
    
    def index_images(self, image_dir):
        """
        为图像目录建立索引
        """
        self.image_paths = []
        features = []
        
        # 遍历图像目录
        for img_path in Path(image_dir).glob('*'):
            if img_path.suffix.lower() in ['.jpg']:
                try:
                    # 读取图像
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                        
                    # 计算特征
                    feature = self.compute_color_histogram(img)
                    
                    # 存储图像路径和特征
                    self.image_paths.append(str(img_path))
                    features.append(feature)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    
        # 建立索引
        for idx, feature in enumerate(features):
            for table_id, projection in enumerate(self.projection_sets):
                hash_val = self._hash_vector(feature, projection)
                self.hash_tables[table_id][hash_val].append((idx, feature))
    
    def query_image(self, query_img_path, k=1):
        """
        查询与输入图像最相似的k张图像
        """
        # 读取查询图像
        query_img = cv2.imread(query_img_path)
        if query_img is None:
            raise ValueError("Cannot read query image")
            
        # 计算查询图像的特征
        query_feature = self.compute_color_histogram(query_img)
        
        # 在哈希表中查找候选项
        candidates = []  # 改用列表而不是集合
        seen_indices = set()  # 用于追踪已经见过的索引
        
        # 在每个哈希表中查找
        for table_id, projection in enumerate(self.projection_sets):
            hash_val = self._hash_vector(query_feature, projection)
            # 获取该哈希值对应的所有候选项
            for idx, feature in self.hash_tables[table_id][hash_val]:
                # 只添加还没见过的索引
                if idx not in seen_indices:
                    candidates.append((idx, feature))
                    seen_indices.add(idx)
                
        if not candidates:
            return []
            
        # 计算距离并排序
        distances = []
        for idx, feature in candidates:
            dist = np.sum((feature - query_feature) ** 2)
            distances.append((dist, idx))
            
        # 返回最相似的k张图像的路径
        distances.sort()
        return [self.image_paths[idx] for _, idx in distances[:k]]
    
    def visualize_results(self, query_img_path, result_paths, save_path=None):
        """
        可视化查询结果
        """
        # 读取并调整查询图像大小
        query_img = cv2.imread(query_img_path)
        query_img = cv2.resize(query_img, (200, 200))
        
        # 读取并调整结果图像大小
        result_imgs = []
        for path in result_paths:
            img = cv2.imread(path)
            img = cv2.resize(img, (200, 200))
            result_imgs.append(img)
            
        # 创建展示图像
        n_results = len(result_imgs)
        display_img = np.zeros((200, 200 * (n_results + 1), 3), dtype=np.uint8)
        
        # 放置查询图像
        display_img[:, :200] = query_img
        
        # 放置结果图像
        for i, img in enumerate(result_imgs):
            display_img[:, (i+1)*200:(i+2)*200] = img
            
        # 显示或保存结果
        if save_path:
            cv2.imwrite(save_path, display_img)
        else:
            cv2.imshow('Query Results', display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if "__main__" == __name__:
    # 定义投影集合
    # index range: 0 ~ 23
    projection_sets = [
        [0, 3, 6, 9, 23, 19],    # 第一个投影集合
        [1, 4, 7, 10, 16, 17],   # 第二个投影集合
        [2, 5, 8, 11, 12, 20]    # 第三个投影集合
    ]

    # 设置路径
    path_to_dir = os.path.dirname(__file__)
    path_to_database = os.path.join(path_to_dir, "img")
    path_to_target = os.path.join(path_to_dir, "target.jpg")

    # 创建LSH实例
    lsh = ImageLSH(projection_sets)

    # 建立图像数据库索引
    image_dir = path_to_database
    lsh.index_images(image_dir)

    # 查询图像
    query_image_path = path_to_target
    results = lsh.query_image(query_image_path, k=1)

    # 可视化结果
    lsh.visualize_results(query_image_path, results)