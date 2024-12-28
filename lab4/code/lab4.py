import os
from Matcher import LSHMatcher, NNMatcher

def analyze_projection_set(projection_sets, database_path, target_path, query_image_path, output_path):
    print("分析投影集合的影响：")
    for i, projection_set in enumerate(projection_sets):
        print(f"第{i+1}个投影集合：")

        # LSH
        lsh = LSHMatcher(projection_set)
        lsh.index_images(database_path)
        results, query_time_LSH = lsh.query_image(query_image_path, k=1)
        lsh.visualize_results(query_image_path, results, save_path=os.path.join(output_path, f"lsh_results_{i+1}.jpg"))

        # NN
        matcher = NNMatcher()
        matcher.index_images(database_path)
        matcher_results, query_time_NN = matcher.query_image(query_image_path, k=1)
        matcher.visualize_results(query_image_path, matcher_results, save_path=os.path.join(output_path, f"nn_results_{i+1}.jpg"))

        print(f"运行时间：")
        print(f"LSH: {query_time_LSH} s")
        print(f"NN: {query_time_NN} s")

        with open(os.path.join(output_path, f"time_projection_set_size.txt"), "a") as f:
            f.write(f"No.{i+1} Projection set: \n")
            f.write(f"LSH: {query_time_LSH} s\n")
            f.write(f"NN: {query_time_NN} s\n")

def analyze_time(projection_set, max_iter, database_path, target_path, query_image_path, output_path):
    print("分析运行时间：")
    LSH_time = 0
    NN_time = 0
    for i in range(max_iter):
        # LSH
        lsh = LSHMatcher(projection_set)
        lsh.index_images(database_path)
        results, query_time_LSH = lsh.query_image(query_image_path, k=1)
        LSH_time += query_time_LSH

        # NN
        matcher = NNMatcher()
        matcher.index_images(database_path)
        matcher_results, query_time_NN = matcher.query_image(query_image_path, k=1)
        NN_time += query_time_NN
    
    LSH_time /= max_iter
    NN_time /= max_iter

    print(f"平均运行时间：")
    print(f"LSH: {LSH_time} s")
    print(f"NN: {NN_time} s")

if "__main__" == __name__:
    # 定义投影集合
    # index range: 0 ~ 23
    projection_sets = [
        [0, 3, 6, 9, 23, 19],    # 第一个投影集合
        [1, 4, 7, 10, 16, 17],   # 第二个投影集合
        [2, 5, 8, 11, 12, 20]    # 第三个投影集合
    ]

    projection_sets_size = [
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        ],
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1],
        ],
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
        ],
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23, 0, 1, 2]
        ],
        [
            [0, 3, 6, 9, 23, 19, 13],
            [1, 4, 7, 10, 16, 17, 14],
            [2, 5, 8, 11, 12, 20, 15]
        ],
        [
            [0, 3, 6, 9, 23, 19],
            [1, 4, 7, 10, 16, 17],
            [2, 5, 8, 11, 12, 20]
        ],
        [
            [0, 3, 6, 9, 23],
            [1, 4, 7, 10, 16],
            [2, 5, 8, 11, 12]
        ],
        [
            [0, 3, 6, 9],
            [1, 4, 7, 10],
            [2, 5, 8, 11]
        ],
        [
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8]
        ],
        [
            [0, 3],
            [1, 4],
            [2, 5]
        ],
        [
            [0],
            [1],
            [2]
        ],
        [
            [0],
            [1]
        ],
        [
            [0]
        ]
    ]

    projection_sets_index = [
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 0]
        ],
        [
            [0, 3, 6, 9, 12],
            [1, 4, 7, 10, 13],
            [2, 5, 8, 11, 14],
            [15, 18, 21, 0, 16],
            [19, 22, 20, 23, 17]
        ],
        [
            [0, 23, 22, 21, 20],
            [19, 18, 17, 16, 15],
            [14, 13, 12, 11, 10],
            [9, 8, 7, 6, 5],
            [4, 3, 2, 1, 0]
        ], 
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 0]
        ],
        [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
            [12, 13],
            [14, 15],
            [16, 17],
            [18, 19],
            [20, 21],
            [22, 23],
        ],
        [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
            [9],
            [10],
            [11],
            [12],
            [13],
            [14],
            [15],
            [16],
            [17],
            [18],
            [19],
            [20],
            [21],
            [22],
            [23],
        ]
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
    # 创建LSH实例
    lsh = LSHMatcher(projection_sets)
    # 建立图像数据库索引
    lsh.index_images(image_dir)
    # 查询图像
    results, query_time_LSH = lsh.query_image(query_image_path, k=1)
    # 可视化结果
    lsh.visualize_results(query_image_path, results, save_path=os.path.join(output, "lsh_results.jpg"))

    # NN
    # 创建Matcher实例
    matcher = NNMatcher()
    # 建立图像数据库索引
    matcher.index_images(image_dir)
    # 查询图像
    matcher_results, query_time_NN = matcher.query_image(query_image_path, k=1)
    # 可视化结果
    matcher.visualize_results(query_image_path, matcher_results, save_path=os.path.join(output, "nn_results.jpg"))

    print("运行时间：")
    with open(os.path.join(output, "time.txt"), "w") as f:
        f.write(f"LSH: {query_time_LSH} s\n")
        f.write(f"NN: {query_time_NN} s\n")
    print(f"LSH: {query_time_LSH} s")
    print(f"NN: {query_time_NN} s")
    print("结果已保存至output文件夹")

    # 分析投影集合的影响
    analyze_projection_set(projection_sets_size, image_dir, path_to_target, query_image_path, output)

    # # 分析运行时间
    # N = 1000
    # analyze_time(projection_sets, N, image_dir, path_to_target, query_image_path, output)