import os
import time
from ImageLSH import ImageLSH
from Matcher import ImageMatcher

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
    image_dir = path_to_database
    query_image_path = path_to_target
    output = os.path.join(path_to_dir, "output")
    os.makedirs(output, exist_ok=True)

    # LSH
    time_LSH_start = time.time()
    # 创建LSH实例
    lsh = ImageLSH(projection_sets)
    # 建立图像数据库索引
    lsh.index_images(image_dir)
    # 查询图像
    results = lsh.query_image(query_image_path, k=1)
    time_LSH_end = time.time()
    # 可视化结果
    lsh.visualize_results(query_image_path, results, save_path=os.path.join(output, "lsh_results.jpg"))

    # NN
    time_NN_start = time.time()
    # 创建Matcher实例
    matcher = ImageMatcher('bf')
    # 建立图像数据库索引
    matcher.index_images(image_dir)
    # 查询图像
    matcher_results = matcher.find_matches(query_image_path, k=1)
    time_NN_end = time.time()
    # 可视化结果
    matcher.visualize_results(query_image_path, matcher_results, save_path=os.path.join(output, "nn_results.jpg"))

    print("运行时间：")
    with open(os.path.join(output, "time.txt"), "w") as f:
        f.write(f"LSH: {time_LSH_end - time_LSH_start} s\n")
        f.write(f"NN: {time_NN_end - time_NN_start} s\n")
    print(f"LSH: {time_LSH_end - time_LSH_start} s")
    print(f"NN: {time_NN_end - time_NN_start} s")
    print("结果已保存至output文件夹")