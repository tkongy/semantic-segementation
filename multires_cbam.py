import keras
from keras.models import *
from keras.layers import *
from keras import layers
from cbam import *
from scSE import *
'''
import tensorflow.keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
'''
IMAGE_ORDERING = "channels_last"
MERGE_AXIS = -1

# 类别对应
matches = [100, 200, 300, 400, 500, 600, 700, 800]


def get_segmentation_model(_input, output):

    img_input = _input
    o = output

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    if IMAGE_ORDERING == 'channels_first':
        output_height = o_shape[2]
        output_width = o_shape[3]
        input_height = i_shape[2]
        input_width = i_shape[3]
        n_classes = o_shape[1]
        o = (Reshape((-1, output_height*output_width)))(o)
        o = (Permute((2, 1)))(o)
    elif IMAGE_ORDERING == 'channels_last':
        output_height = o_shape[1]
        output_width = o_shape[2]
        input_height = i_shape[1]
        input_width = i_shape[2]
        n_classes = o_shape[3]
        o = (Reshape((output_height*output_width, -1)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)

    return model


def mlti_res_block(inputs, filter_size1, filter_size2, filter_size3, filter_size4):
    cnn1 = Conv2D(filter_size1, (3, 3), padding='same', activation="relu")(inputs)
    cnn1 = (BatchNormalization())(cnn1)
    cnn2 = Conv2D(filter_size2, (3, 3), padding='same', activation="relu")(cnn1)
    cnn2 = (BatchNormalization())(cnn2)
    cnn3 = Conv2D(filter_size3, (3, 3), padding='same', activation="relu")(cnn2)
    cnn3 = (BatchNormalization())(cnn3)

    cnn = Conv2D(filter_size4, (1, 1), padding='same', activation="relu")(inputs)
    cnn = (BatchNormalization())(cnn)

    concat = Concatenate()([cnn1, cnn2, cnn3])
    add = Add()([concat, cnn])

    return add

def res_path(inputs, filter_size, path_number):
    def block(x, fl):
        cnn1 = Conv2D(fl, (3, 3), padding='same', activation="relu")(x)
        cnn1 = (BatchNormalization())(cnn1)
        cnn2 = Conv2D(fl, (1, 1), padding='same', activation="relu")(x)
        cnn2 = (BatchNormalization())(cnn2)

        add = Add()([cnn1, cnn2])

        return add
    cnn = inputs
    # cnn = block(inputs, filter_size)
    if path_number < 5:
        cnn = block(cnn, filter_size)
        if path_number < 4:
            cnn = block(cnn, filter_size)
            if path_number < 3:
                cnn = block(cnn, filter_size)
                if path_number < 2:
                    cnn = block(cnn, filter_size)

    return cnn


def multi_res_u_net(pretrained_weights=None, input_height=256,  input_width=256, lr=0.001, n_classes=8):

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    inputs = img_input

    res_block1 = mlti_res_block(inputs, 8, 17, 26, 51)
    pool1 = MaxPool2D()(res_block1)

    res_block2 = mlti_res_block(pool1, 17, 35, 53, 105)
    pool2 = MaxPool2D()(res_block2)

    res_block3 = mlti_res_block(pool2, 31, 72, 106, 209)
    pool3 = MaxPool2D()(res_block3)

    res_block4 = mlti_res_block(pool3, 71, 142, 213, 426)
    pool4 = MaxPool2D()(res_block4)

    res_block5 = mlti_res_block(pool4, 142, 284, 427, 853)
    upsample = UpSampling2D()(res_block5)
    inputs = upsample
    residual = Conv2D(853, (1, 1), padding='same', activation="relu")(inputs)
    residual = (BatchNormalization())(residual)
    # cbam = cbam_block(residual)
    cbam = scse_block(residual)
    upsample = layers.add([upsample, residual, cbam])

    res_path4 = res_path(res_block4, 256, 4)
    concat = Concatenate()([upsample, res_path4])

    res_block6 = mlti_res_block(concat, 71, 142, 213, 426)
    upsample = UpSampling2D()(res_block6)
    inputs = upsample
    residual = Conv2D(426, (1, 1), padding='same', activation="relu")(inputs)
    residual = (BatchNormalization())(residual)
    #cbam = cbam_block(residual)
    cbam = scse_block(residual)
    upsample = layers.add([upsample, residual, cbam])

    res_path3 = res_path(res_block3, 128, 3)
    concat = Concatenate()([upsample, res_path3])

    res_block7 = mlti_res_block(concat, 31, 72, 106, 209)
    upsample = UpSampling2D()(res_block7)
    inputs = upsample
    residual = Conv2D(209, (1, 1), padding='same', activation="relu")(inputs)
    residual = (BatchNormalization())(residual)
    #cbam = cbam_block(residual)
    cbam = scse_block(residual)
    upsample = layers.add([upsample, residual, cbam])

    res_path2 = res_path(res_block2, 64, 2)
    concat = Concatenate()([upsample, res_path2])

    res_block8 = mlti_res_block(concat, 17, 35, 53, 105)
    upsample = UpSampling2D()(res_block8)
    inputs = upsample
    residual = Conv2D(105, (1, 1), padding='same', activation="relu")(inputs)
    residual = (BatchNormalization())(residual)
    #cbam = cbam_block(residual)
    cbam = scse_block(residual)
    upsample = layers.add([upsample, residual, cbam])

    res_path1 = res_path(res_block1, 32, 1)
    concat = Concatenate()([upsample, res_path1])

    res_block9 = mlti_res_block(concat, 8, 17, 26, 51)
    sigmoid = Conv2D(n_classes, (1, 1), padding='same')(res_block9)      # 输出类别为8 将filter_size由1改为8
    model = get_segmentation_model(img_input, sigmoid)

    return model