import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers

def resnet_block(inputs, num_filters, kernel_size=3, stride=1, conv_first=True):
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=stride, padding='same',
                  kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    else:
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = conv(x)
    return x

def resnet18(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('Depth should be 6n+2 (e.g. 20, 32, 44).')
    num_filters = 64
    num_res_blocks = int((depth - 2) / 6)

    inputs = layers.Input(shape=input_shape)
    x = resnet_block(inputs, num_filters)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # First layer but not first stack
                strides = 2  # downsample
            y = resnet_block(x, num_filters, stride=strides)
            y = resnet_block(y, num_filters, conv_first=False)
            if stack > 0 and res_block == 0:  # First layer but not first stack
                x = Conv2D(num_filters, kernel_size=1, strides=strides, padding='same',
                           kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            x = layers.add([x, y])
        num_filters *= 2

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
