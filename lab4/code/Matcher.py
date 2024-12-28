import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

class LSHMatcher:
    def __init__(self, projection_sets):
        """
        初始化LSH \\
        projection_sets: List[ List[ int]], 投影集合的列表
        """
        self.projection_sets = projection_sets
        self.hash_tables = [defaultdict(list) for _ in range(len(projection_sets))]
        
    def compute_color_histogram(self, image):
        """
        计算12维颜色直方图特征
        将图像分为4个区域，每个区域计算3维颜色直方图 \\
        INPUT:
            image: np.ndarray, 输入图像
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
        将特征向量量化为0,1,2三个值 \\
        INPUT:
            feature: np.ndarray, 特征向量
        """
        quantized = np.zeros_like(feature)
        quantized[feature > 0.6] = 2
        quantized[(feature >= 0.3) & (feature <= 0.6)] = 1
        return quantized
    
    def _hash_vector(self, feature, projection):
        """
        计算特征向量在某个投影集合上的哈希值 \\
        INPUT:
            feature: np.ndarray, 特征向量
            projection: List[ int], 投影集合
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
        为图像目录建立索引 \\
        INPUT:
            image_dir: str, 图像目录路径
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
        查询与输入图像最相似的k张图像 \\
        INPUT:
            query_img_path: str, 查询图像路径
            k: int, 返回的匹配图像数量
        OUTPUT:
            List[ str], 最相似的k张图像的路径
            search_time: float, 查询时间
        """
        # 记录开始时间
        start_time = time.time()

        # 读取查询图像
        query_img = cv2.imread(query_img_path)
        if query_img is None:
            raise ValueError("Cannot read query image")
            
        # 计算查询图像的特征
        query_feature = self.compute_color_histogram(query_img)
        
        # 在哈希表中查找候选项
        candidates = [] 
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

        # 记录结束时间
        end_time = time.time()
        search_time = end_time - start_time

        return [self.image_paths[idx] for _, idx in distances[:k]], search_time
    
    def visualize_results(self, query_img_path, result_paths, save_path=None):
        """
        可视化查询结果 \\
        INPUT:
            query_img_path: str, 查询图像路径
            result_paths: List[ str], 结果图像路径列表
            save_path: str, 可选, 保存图像路径
        """
        # 读取并调整查询图像大小
        query_img = cv2.imread(query_img_path)
        
        # 读取并调整结果图像大小
        result_imgs = []
        for path in result_paths:
            img = cv2.imread(path)
            result_imgs.append(img)
        
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        plt.title('Query Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_imgs[0], cv2.COLOR_BGR2RGB))
        plt.title('Best Match')
        plt.axis('off')
        
        plt.suptitle('Results For LSH')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

class NNMatcher:
    def __init__(self):
        self.image_paths = []
        self.features = []
        
    def compute_color_histogram(self, image):
        """
        计算12维颜色直方图特征
        将图像分为4个区域，每个区域计算3维颜色直方图 \\
        INPUT:
            image: np.ndarray, 输入图像
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

    def index_images(self, image_dir):
        """
        为图像目录建立索引
        """
        image_dir = Path(image_dir)
        
        for img_path in image_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg']:
                try:
                    # 读取图像
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # 计算特征
                    feature = self.compute_color_histogram(img)
                    
                    self.image_paths.append(str(img_path))
                    self.features.append(feature)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # 将特征列表转换为numpy数组，便于后续计算
        self.features = np.array(self.features)
        
    def query_image(self, query_img_path, k=1):
        """
        使用NN算法查找最相似的k张图像
        """
        # 记录开始时间
        start_time = time.time()
        
        # 读取查询图像
        query_img = cv2.imread(query_img_path)
        if query_img is None:
            raise ValueError("Cannot read query image")
            
        # 计算查询图像的特征
        query_feature = self.compute_color_histogram(query_img)
        
        # 计算与所有图像的距离
        distances = np.sum((self.features - query_feature) ** 2, axis=1)
        
        # 获取前k个最小距离的索引
        nearest_indices = np.argsort(distances)[:k]
        
        # 记录结束时间
        end_time = time.time()
        search_time = end_time - start_time
        
        # 返回最相似图像的路径和搜索时间
        return [self.image_paths[i] for i in nearest_indices], search_time
    
    def visualize_results(self, query_img_path, result_paths, save_path=None):
        """
        可视化查询结果 \\
        INPUT:
            query_img_path: str, 查询图像路径
            result_paths: List[ str], 结果图像路径列表
            save_path: str, 可选, 保存图像路径
        """
        # 读取并调整查询图像大小
        query_img = cv2.imread(query_img_path)
        
        # 读取并调整结果图像大小
        result_imgs = []
        for path in result_paths:
            img = cv2.imread(path)
            result_imgs.append(img)
        
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        plt.title('Query Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_imgs[0], cv2.COLOR_BGR2RGB))
        plt.title('Best Match')
        plt.axis('off')
        
        plt.suptitle('Results For NN')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()