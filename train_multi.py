from dataloader import image_segmentation_generator
from MultiRes_Unet import *
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.metrics import categorical_crossentropy

# 使用CPU
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def train(model, image_folder, label_folder, n_class, batch_size=2, epochs=20, weights_path=None):
    # model：传入模型
    # image_folder：图像文件夹
    # label_folder：分割数据文件夹
    # n_class：类别数量
    # weights_path：模型权重路径
    train_gen = image_segmentation_generator(
        image_folder, label_folder, batch_size, n_class)
    nadam = optimizers.Nadam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss='categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['categorical_accuracy'])

    if weights_path != None:
         model.load_weights(weights_path)

   # model.fit_generator(train_gen, 200, epochs=epochs)
    checkpointer = ModelCheckpoint(filepath="weight.h5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
    model.fit_generator(train_gen, 100, epochs=epochs, callbacks=[checkpointer])

def save_model(model, model_path_last, model1_path):
    model.save_weights(model_path_last)
    model.save(model1_path)



if __name__ == "__main__":
    weights_path_last = "weight_last.h5"
    weights_path="weight.h5"
    model1_path="model.h5"
    image_folder = "D:\\python file\\AI Remote Sensing Image\\baseline\\train\\images/"
    label_folder = "D:\\python file\\AI Remote Sensing Image\\baseline\\train\\labels/"
    n_class = 8

    model = multi_res_u_net()
    train(model, image_folder, label_folder,
          n_class, weights_path=None)
    save_model(model, model_path_last=weights_path_last, model1_path=model1_path)
