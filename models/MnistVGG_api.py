import numpy as np
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D,
                          MaxPooling2D, Activation, Dense, Dropout, Flatten)
from tensorflow.keras.models import Model

from DeepEverest import DeepEverest


class MnistVGG(DeepEverest):
    """
    1. ZeroPadding2D (2, 2)
    2. (3X3 Conv2D 64) X 2 + maxpool
    3. (3X3 Conv2D 128) X 2 + maxpool
    4. (3X3 Conv2D 256) X 3 + maxpool
    5. (3X3 Conv2D 512) X 3 + maxpool
    6. (3X3 Conv2D 512) X 3 + maxpool
    7. (FC 256 + Dropout(0.5)) X 2
    8. FC 10 + Softmax
    """

    def __init__(self, lib_file, dataset, train=False):
        model=self._build()
        DeepEverest.__init__(self, model, False, lib_file, dataset, bs=64)
        if not train:
            self.model.load_weights('./models/mnistvgg.h5')


    @staticmethod
    def _build():
        """
        Builds MnistVGG. Details written in the paper below.
        - Very Deep Convolutional Networks for Large-Scale Image Recognition
          (https://arxiv.org/abs/1409.1556)

        Returns:
            MnistVGG model
        """

        x = Input(shape=(28, 28, 1))
        y = ZeroPadding2D(padding=(2, 2))(x)  # matching the image size of CIFAR-10

        y = MnistVGG._multi_conv_pool(y, 64, 2)  # 32x32
        y = MnistVGG._multi_conv_pool(y, 128, 2)  # 16x16
        y = MnistVGG._multi_conv_pool(y, 256, 3)  # 8x8
        y = MnistVGG._multi_conv_pool(y, 512, 3)  # 4x4
        y = MnistVGG._multi_conv_pool(y, 512, 3)  # 2x2
        y = Flatten()(y)
        y = Dense(units=256, activation='relu')(y)  # original paper suggests 4096 FC
        y = Dropout(0.5)(y)
        y = Dense(units=256, activation='relu')(y)
        y = Dropout(0.5)(y)
        y = Dense(units=10)(y)
        y = Activation('softmax')(y)

        return Model(x, y, name='MnistVGG')

    @staticmethod
    def _multi_conv_pool(x, output_channel, n):
        """
        Builds (Conv2D - BN - Relu) X n - MaxPooling2D
        The training is regularized by global weight decay (5e-4) in the original paper,
        but BN is used here instead of weight decay

        Returns:
            multi conv + max pooling block
        """

        y = x
        for _ in range(n):
            y = Conv2D(output_channel, (3, 3), padding='same')(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
        y = MaxPooling2D(strides=(2, 2))(y)
        return y

    def preprocess_input_for_inference(self, x):
        h, w = 28, 28
        x = np.reshape(x, (-1, h * w))
        numerator = x - np.expand_dims(np.mean(x, 1), 1)
        denominator = np.expand_dims(np.std(x, 1), 1)
        return np.reshape(numerator / (denominator + 1e-7), (-1, h, w, 1))
