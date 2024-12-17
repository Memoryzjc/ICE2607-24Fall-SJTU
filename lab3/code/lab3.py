import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def build_image_pyramid(image, levels, scale_factor=0.5):
    """优化图像金字塔构建"""
    pyramid = [image]
    current_image = image
    for i in range(1, levels):
        # 使用 cv2.resize 进行下采样
        new_width = int(current_image.shape[1] * scale_factor)
        new_height = int(current_image.shape[0] * scale_factor)
        current_image = cv2.resize(current_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        pyramid.append(current_image)

    return pyramid

def non_max_suppression(response, window_size=3):
    suppressed = np.zeros_like(response)
    for i in range(window_size // 2, response.shape[0] - window_size // 2):
        for j in range(window_size // 2, response.shape[1] - window_size // 2):
            window = response[i - window_size // 2:i + window_size // 2 + 1, j - window_size // 2:j + window_size // 2 + 1]
            if response[i, j] == np.max(window):
                suppressed[i, j] = response[i, j]
    return suppressed

def multi_scale_harris_corner_detection(image, levels=3, scale_factor=0.5, block_size=2, ksize=3, k=0.1, threshold_ratio=0.01):
    """使用图像金字塔进行多尺度Harris角点检测"""
    pyramid = build_image_pyramid(image, levels, scale_factor=scale_factor)
    all_keypoints = []
    for level, img in enumerate(pyramid):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, blockSize=block_size, ksize=ksize, k=k)
        # dst = cv2.dilate(dst, None)
        # dst = non_max_suppression(dst)
        threshold = threshold_ratio * dst.max()
        keypoints = np.argwhere(dst > threshold)
        for pt in keypoints:
            kp = cv2.KeyPoint(float(pt[1] * ((1 / scale_factor) ** level)), float(pt[0] * ((1 / scale_factor) ** level)), 1 * ((1 / scale_factor) ** level))
            all_keypoints.append(kp)
    print("Finish multi-scale Harris corner detection")
    return all_keypoints

def bilinear_interpolate(image, x, y):
    """双线性插值"""
    # 确保目标位置在图像范围内
    x1 = max(int(np.floor(x)), 0)
    x1 = min(x1, image.shape[1] - 1)
    y1 = max(int(np.floor(y)), 0)
    y1 = min(y1, image.shape[0] - 1)

    x2 = min(x1 + 1, image.shape[1] - 1)
    y2 = min(y1 + 1, image.shape[0] - 1)

    # 获取四个邻近点的像素值
    Q11, Q21 = image[y1, x1], image[y1, x2]
    Q12, Q22 = image[y2, x1], image[y2, x2]

    # 计算插值
    dx1, dy1 = x - x1, y - y1
    dx2, dy2 = x2 - x, y2 - y
    P = Q11 * dx2 * dy2 + Q21 * dx1 * dy2 + Q12 * dx2 * dy1 + Q22 * dx1 * dy1

    return P

def compute_sift_descriptors(image, keypoints):
    """自己实现SIFT描述子的计算"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    gray = np.float32(gray)
    shape = gray.shape

    descriptors = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        # 提取关键点邻域的梯度信息
        window = gray[max(0, y - 8): min(y + 8, shape[0]), max(0, x - 8): min(x + 8, shape[1])]
        if window.shape[0] < 16 or window.shape[1] < 16:
            continue

        # 计算梯度
        dx = cv2.Sobel(window, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(window, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(dx**2 + dy**2)
        angle = np.rad2deg(np.arctan2(dy, dx)) % 360

        # 计算关键点的主方向
        hist = np.zeros(36, dtype=np.float32)
        for i in range(magnitude.shape[0]):
            for j in range(magnitude.shape[1]):
                bin_idx = int(angle[i, j] // 10) % 36
                hist[bin_idx] += magnitude[i, j]
        main_orientation = np.argmax(hist) * 10
        obj_angle = angle[8, 8] - main_orientation

        # 构建SIFT描述子
        descriptor = []
        for i in range(0, 16, 4):
            for j in range(0, 16, 4):
                block_hist = np.zeros(8, dtype=np.float32)
                for m in range(4):
                    for n in range(4):
                        # 物体坐标系上的点
                        obj_x = j + n
                        obj_y = i + m

                        # 图像坐标系上的点
                        img_x = obj_x * np.cos(np.deg2rad(obj_angle)) - obj_y * np.sin(np.deg2rad(obj_angle))
                        img_y = obj_x * np.sin(np.deg2rad(obj_angle)) + obj_y * np.cos(np.deg2rad(obj_angle))

                        # 双线性插值
                        mag = bilinear_interpolate(magnitude, img_x, img_y)
                        ang = bilinear_interpolate(angle, img_x, img_y)

                        # 临近插值
                        # y1 = max(int(np.floor(img_y)), 0)
                        # y1 = min(y1, magnitude.shape[0] - 1)
                        # x1 = max(int(np.floor(img_x)), 0)
                        # x1 = min(x1, magnitude.shape[1] - 1)
                        # mag = magnitude[y1, x1]
                        # ang = angle[y1, x1]

                        ang = (ang + 360) % 360
                        bin_idx = int(ang // 45) % 8
                        block_hist[bin_idx] += mag
                descriptor.extend(block_hist)
        
        descriptor = np.array(descriptor, dtype=np.float32)
        descriptor /= np.linalg.norm(descriptor) + 1e-6
        descriptors.append(descriptor)

    print("Finish SIFT descriptor computation")
    return np.array(descriptors)

def match_features(descriptors1, descriptors2, ratio_threshold=0.75):
    matches = []
    matched = [False] * len(descriptors2)

    for i, desc1 in enumerate(descriptors1):
        best_match = None
        best_distance = float('inf')
        second_best_distance = float('inf')
        for j, desc2 in enumerate(descriptors2):
            if matched[j]:
                continue
            distance = np.linalg.norm(desc1 - desc2)
            if distance < best_distance:
                second_best_distance = best_distance
                best_distance = distance
                best_match = j
        if best_distance < ratio_threshold * second_best_distance:
            matches.append((i, best_match, best_distance))
            matched[best_match] = True

    print("Finish feature matching")
    return matches

def match_features_BFMatcher(descriptors1, descriptors2, ratio_threshold=0.75):
    """使用BFMatcher进行特征匹配"""
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # 使用KNN进行匹配，k=2返回两个最佳匹配
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        # 比率测试：距离最小的匹配与第二小匹配的距离比率
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    print("Finish feature matching")
    return good_matches

def main(ratio_threshold=0.75, levels=3, scale_factor=0.3, block_size=2, ksize=3, k=0.04, threshold_ratio=0.01):
    cwd = os.path.dirname(__file__)
    # 读取目标图像
    target_image = cv2.imread(os.path.join(cwd, 'img', 'target.jpg'))
    if target_image is None:
        print("无法读取目标图像")
        return

    # 提取目标图像的特征点和描述子
    target_keypoints = multi_scale_harris_corner_detection(target_image, levels=levels, scale_factor=scale_factor, block_size=block_size, ksize=ksize, k=k, threshold_ratio=threshold_ratio)
    target_descriptors = compute_sift_descriptors(target_image, target_keypoints)

    best_match_image = None
    best_match_keypoints = None
    best_match_descriptors = None
    best_matches = []
    best_score = 0

    # 遍历数据集中的所有图片
    for i in range(1, 6):  # 数据集中有5张图片
        search_image = cv2.imread(os.path.join(cwd, 'img', f'{i}.jpg'))
        if search_image is None:
            print(f"无法读取 {i}.jpg")
            continue

        # 提取搜索图像的特征点和描述子
        search_keypoints = multi_scale_harris_corner_detection(search_image, levels=levels, block_size=block_size, ksize=ksize, k=k, threshold_ratio=threshold_ratio)
        search_descriptors = compute_sift_descriptors(search_image, search_keypoints)

        # 匹配特征点
        # 平均距离作为分数
        matches = match_features(target_descriptors, search_descriptors, ratio_threshold=ratio_threshold)

        # 匹配特征点-使用BFMatcher
        # matches = match_features_BFMatcher(target_descriptors, search_descriptors, ratio_threshold=ratio_threshold)
        # 匹配的数量作为分数
        score = len(matches)
        if score > best_score:
            best_score = score
            best_match_image = search_image
            best_match_keypoints = search_keypoints
            best_match_descriptors = search_descriptors
            best_matches = matches
        print(score)

    # 绘制最佳匹配结果
    if best_match_image is not None:
        result_image = cv2.drawMatches(
            best_match_image, best_match_keypoints,
            target_image, target_keypoints,
            [cv2.DMatch(_queryIdx=m[1], _trainIdx=m[0], _imgIdx=0, _distance=m[2]) for m in best_matches],
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # 使用Matplotlib显示图像
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title("Best Match")
        plt.axis('off')
        plt.savefig(os.path.join(cwd, 'output', 'my_match.png'))
        plt.close()
    else:
        print("未找到匹配的图像")

def sift_feature_detection(image_path, target_image_path, output_path, ratio_threshold=0.75):
    """使用OpenCV的SIFT进行特征检测和匹配"""
    # 读取图像
    image = cv2.imread(image_path)
    target_image = cv2.imread(target_image_path)
    
    if image is None or target_image is None:
        print("无法读取图像")
        return

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # 创建SIFT对象
    sift = cv2.SIFT_create()

    # 检测关键点并计算描述子
    keypoints_image, descriptors_image = sift.detectAndCompute(gray_image, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(gray_target_image, None)

    # 使用暴力匹配器进行匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # KNN匹配
    matches = bf.knnMatch(descriptors_image, descriptors_target, k=2)

    # 使用比率测试来过滤不好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # 绘制匹配结果
    result_image = cv2.drawMatches(
        image, keypoints_image,
        target_image, keypoints_target,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 显示匹配结果
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("SIFT Feature Matching")
    plt.axis('off')
    plt.savefig(os.path.join(output_path, 'sift_matching.png'))
    plt.close()

    return good_matches, keypoints_image, keypoints_target

def contrast_main():
    cwd = os.path.dirname(__file__)
    # 设置目标图像和查询图像的路径
    image_path = os.path.join(cwd, 'img', '3.jpg')  
    target_image_path = os.path.join(cwd, 'img', 'target.jpg') 
    output_path = os.path.join(cwd, 'output')
    
    # 调用SIFT特征检测与匹配函数
    good_matches, keypoints_image, keypoints_target = sift_feature_detection(image_path, target_image_path, output_path)

    # 打印匹配的数量
    print(f"匹配的特征点数量: {len(good_matches)}")

if __name__ == "__main__":
    cwd = os.path.dirname(__file__)
    os.makedirs(os.path.join(cwd, 'output'), exist_ok=True)
    main()
    contrast_main()