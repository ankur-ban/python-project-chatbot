import tensorflow as tf
from tensorflow.keras import layers, models, backend as K


def conv_block(x, growth_rate, name):
    """A building block for a dense layer with bottleneck."""
    x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_bn')(x)
    x1 = layers.Activation('relu', name=name + '_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_conv1')(x1)

    x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_bn2')(x1)
    x1 = layers.Activation('relu', name=name + '_relu2')(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_conv2')(x1)

    x = layers.Concatenate(axis=-1, name=name + '_concat')([x, x1])
    return x


def dense_block(x, blocks, growth_rate, name):
    """A dense block is a stack of convolutional blocks."""
    for i in range(blocks):
        x = conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """Transition layers appear between dense blocks to reduce feature map size."""
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(K.int_shape(x)[-1] * reduction), 1, use_bias=False, name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def DenseNet(blocks, growth_rate=32, include_top=True, weights=None,
             input_shape=(224, 224, 3), classes=1000, reduction=0.5):
    """
    DenseNet model builder

    Args:
    - blocks: list of integers, number of conv blocks in each dense block (e.g. [6,12,64,48] for DenseNet264)
    - growth_rate: how many filters to add per conv block
    - include_top: whether to include classification head
    - weights: pre-trained weights path or None
    - input_shape: input image shape
    - classes: number of classes for classification
    - reduction: compression factor in transition layers

    Returns:
    - Keras Model
    """
    img_input = layers.Input(shape=input_shape)

    # Initial convolution and pooling
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(2 * growth_rate, 7, strides=2, use_bias=False, name='conv1_conv')(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    # Build dense blocks and transition layers
    for i in range(len(blocks)):
        x = dense_block(x, blocks[i], growth_rate, name='conv' + str(i + 2))
        if i != len(blocks) - 1:
            x = transition_block(x, reduction, name='pool' + str(i + 2))

    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc')(x)

    model = models.Model(img_input, x, name='densenet')

    if weights:
        model.load_weights(weights)

    return model
