from model_train_para import model_para
from keras import layers


class model_select(model_para):

    def default_model(self, model):
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        return model

    def VGG16(self):
        from keras.applications import VGG16
        conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=self.input_shape)
        return conv_base

    def IncetionResNetV2(self):  # OOM..............ORZ
        # Input size must be at least 139x139
        from keras.applications import InceptionResNetV2
        conv_base = InceptionResNetV2(include_top=False,
                                      weights='imagenet',
                                      input_shape=self.input_shape)
        return conv_base

    def InceptionV3(self):  # OOM..............ORZ
        from keras.applications import InceptionV3
        conv_base = InceptionV3(include_top=False,
                                weights='imagenet',
                                input_shape=self.input_shape)
        return conv_base

    def MobileNet(self):
        from keras.applications import MobileNet
        alpha = 0.5
        conv_base = MobileNet(include_top=False,
                              weights='imagenet',
                              input_shape=self.input_shape,
                              pooling='max',
                              alpha=alpha)
        '''
        alpha: 控制网络的宽度：
                  如果alpha<1，则同比例的减少每层的滤波器个数
                  如果alpha>1，则同比例增加每层的滤波器个数
                  如果alpha=1，使用默认的滤波器个数
        '''
        return conv_base, alpha

    def fine_tune_layers(self, trainable_layer, conv_base):
        conv_base.trainable = True
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == trainable_layer:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
        return conv_base
