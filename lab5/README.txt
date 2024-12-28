代码文件测试方法：

5.1 Pytorch

代码位于 code/Pytorch 中，进入 code/Pytorch 目录下：

1. 在控制台中输入 `python exp2.py` 即可测试训练模型文件，使用 GPU 训练大概 20min 左右即可在 ./checkpoint 目录下得到最终训练好的模型。

2. 在控制台中输入 `python draw.py` 即可测试绘制变化趋势文件，保存在 Pytorch/test_acc.png 文件。如需修改，请修改文件中 `test_acc` 列表。

5.2 CNN

代码位于 code/CNN 中，进入 code/CNN 目录下：

1. 在控制台中输入 `python main.py` 即可测试主代码，输入数据集图像位于 CNN/Dataset 中，输入待检索图像位于 CNN/Query_image 中，输出文件夹分别为：CNN/Feature_xxx --- 使用 xxx 模型得到的数据集图像特征；CNN/Query_feature --- 待检索图像的特征；CNN/Output --- 检索结果。

2. 如果想要测试每个模型的特征提取效果，请在 `extract_feature_xxx.py` 文件中自行加入测试代码进行测试。