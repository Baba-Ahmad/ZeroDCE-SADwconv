from tensorflow import keras
from tensorflow.keras import layers

def build_dce_net():
    input_img = keras.Input(shape=[None, None, 3])
    conv1 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(input_img)
    conv2 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv1)
    conv3 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv2)
    conv4 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv3)

    # Depthwise Convolution
    conv4_dw = layers.DepthwiseConv2D(
        (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv4)

    # Self-Attention
    att_con1 = layers.Concatenate(axis=-1)([conv4, conv4_dw])
    att_gap = layers.GlobalAveragePooling2D()(att_con1)
    att_gap = layers.Reshape((1, 1, 64))(att_gap)
    att_conv1 = layers.Conv2D(
        4, (1, 1), strides=(1, 1), activation="relu", padding="same"
    )(att_gap)
    att_conv2 = layers.Conv2D(
        64, (1, 1), strides=(1, 1), activation="sigmoid", padding="same"
    )(att_conv1)
    att_mul = layers.Multiply()([att_con1, att_conv2])

    int_con1 = layers.Concatenate(axis=-1)([att_mul, conv3])
    conv5 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(int_con2)
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])

    x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(
        int_con3
    )
    return keras.Model(inputs=input_img, outputs=x_r)
