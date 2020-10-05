# semantic-segementation
Remote sensing image semantic segmentation tf2-multiresunet
基于全国人工智能大赛的遥感赛道提供的baseline修改而得
# 源代码在Windows + CPU环境下使用，
# 若要使用GPU，可将train/predict中强制使用CPU的相关代码注释

# 文件说明
训练集保存在./train中，其中
(1)./train/images存放训练图像
(2)./train/labels存放人工标记

测试集保存在./test中，其中
(1)./test/images存放测试图像
(2)./test/labels存放人工标记

weight.h5用来保存训练的网络权重

# update
增加了cbam模块

