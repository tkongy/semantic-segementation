
import random
import os
import itertools
import cv2
import numpy as np
from model import matches
import matplotlib.pyplot as plt


def get_img_label_paths(images_path, labels_path):
    res = []
    for dir_entry in os.listdir(images_path):            # dir_entry遍历列表 os.lister读取指定路径下文件和文件夹的列表
        if os.path.isfile(os.path.join(images_path, dir_entry)):
            # os.path.isfile用于判断某一对象(需提供绝对路径)是否为文件 os.path.join路径拼接
            file_name, _ = os.path.splitext(dir_entry)  # os.path.splitext()分离文件名和拓展名
            res.append((os.path.join(images_path, file_name+".tif"),
                        os.path.join(labels_path, file_name+".png")))
    return res


def get_image_array(img):
    return np.float32(img) / 127.5 - 1


def get_segmentation_array(img, nClasses):
    seg_labels = np.zeros((256, 256, nClasses))

    for m in matches:
        img[img == m] = matches.index(m)

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    seg_labels = np.reshape(seg_labels, (256*256, nClasses))
    return seg_labels

def image_segmentation_generator(images_path, labels_path, batch_size, num_class):
    img_seg_pairs = get_img_label_paths(images_path, labels_path)
    random.shuffle(img_seg_pairs)  # 打乱列表顺序
    zipped = itertools.cycle(img_seg_pairs)   # 遍历元素并产生重复循环
    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            img, seg = next(zipped)

            img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            seg = cv2.imread(seg, cv2.IMREAD_UNCHANGED)   # cv2.IMREAD_UNCHANGED：包括alpha 可以完整读取

            X.append(get_image_array(img))
            Y.append(get_segmentation_array(seg, num_class))

        yield np.array(X), np.array(Y)


# if __name__ == "__main__":
#     label = "C:/test2/result/46_0.png"
#     seg = cv2.imread(label, cv2.IMREAD_UNCHANGED)
#     # for m in matches:
#     #     seg[seg == m] = matches.index(m)
#     # save_file = "C:/Users/zze/Desktop/wtfk.png"
#     # cv2.imwrite(save_file,seg)
#     plt.imshow(np.array(seg))
#     plt.show()
