from model_train_para import model_para
from keras import optimizers
from keras import models
from keras import regularizers
from keras import layers


class model_select(model_para):

    def default_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        if len(self.labels) == 2:  # 2分类
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer=self.optim,
                          metrics=['acc'])
        else:  # 多分类
            model.add(layers.Dense(len(self.labels), activation='softmax'))
            # model.compile(loss='categorical_crossentropy',
            #               optimizer=optimizers.RMSprop(lr=2e-3),
            #               metrics=['acc'])
            #sgd = optimizers.SGD(lr=1e-3, momentum=0.9, decay=1e-3, nesterov=False)
            model.compile(loss='categorical_crossentropy',
                          optimizer=self.optim,
                          metrics=['acc'])
        return model

    def VGG16(self):
        from keras.applications import VGG16
        model = models.Sequential()
        conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=self.input_shape)
        print('VGG16架构:\n')
        conv_base.summary()

        """
        pay attention that this is not the standard structure of vgg16.
        official top structure:
                    x = Flatten(name='flatten')(x)
                    x = Dense(4096, activation='relu', name='fc1')(x)
                    x = Dense(4096, activation='relu', name='fc2')(x)
                    x = Dense(classes, activation='softmax', name='predictions')(x)

        """
        conv_base = self.fine_tune_layers('block5_conv1', conv_base)
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu',
                               kernel_regularizer=regularizers.l2(0.01)))
        if len(self.labels) == 2:  # 2分类
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer=self.optim,
                          metrics=['acc'])
        else:  # 多分类
            model.add(layers.Dense(len(self.labels), activation='softmax'))
            #sgd = optimizers.SGD(lr=1e-3, momentum=0.9, decay=1e-3, nesterov=False)
            model.compile(loss='categorical_crossentropy',
                          optimizer=self.optim,
                          metrics=['acc'])
        return model

    def IncetionResNetV2(self):  # OOM..............ORZ
        # Input size must be at least 139x139
        model = models.Sequential()
        from keras.applications import InceptionResNetV2
        conv_base = InceptionResNetV2(include_top=False,
                                      weights='imagenet',
                                      input_shape=self.input_shape)
        print('IncetionResNetV2:\n')
        conv_base.summary()
        model.add(conv_base)
        model.add(layers.Flatten())
        if self.extra_dense_layers:
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(512, activation='relu',
                                   kernel_regularizer=regularizers.l2(0.01)))
        if len(self.labels) == 2:  # 2分类
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer=self.optim,
                          metrics=['acc'])
        else:  # 多分类
            model.add(layers.Dense(len(self.labels), activation='softmax'))
            # model.compile(loss='categorical_crossentropy',
            #               optimizer=optimizers.RMSprop(lr=2e-3),
            #               metrics=['acc'])
            #sgd = optimizers.SGD(lr=1e-3, momentum=0.9, decay=1e-3, nesterov=False)
            model.compile(loss='categorical_crossentropy',
                          optimizer=self.optim,
                          metrics=['acc'])
        return model

    def InceptionV3(self):  # OOM..............ORZ
        model = models.Sequential()
        from keras.applications import InceptionV3
        conv_base = InceptionV3(include_top=False,
                                weights='imagenet',
                                input_shape=self.input_shape)
        print('InceptionV3:\n')
        conv_base.summary()
        model.add(conv_base)
        model.add(layers.Flatten())
        if self.extra_dense_layers:
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(512, activation='relu',
                                   kernel_regularizer=regularizers.l2(0.01)))
        if len(self.labels) == 2:  # 2分类
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer=self.optim,
                          metrics=['acc'])
        else:  # 多分类
            model.add(layers.Dense(len(self.labels), activation='softmax'))
            # model.compile(loss='categorical_crossentropy',
            #               optimizer=optimizers.RMSprop(lr=2e-3),
            #               metrics=['acc'])

            model.compile(loss='categorical_crossentropy',
                          optimizer=self.optim,
                          metrics=['acc'])
        return model

    def MobileNet(self):
        model = models.Sequential()
        from keras.applications import MobileNet
        alpha = 1
        conv_base = MobileNet(include_top=False,
                              weights="imagenet",
                              input_shape=self.input_shape,
                              pooling='max',
                              alpha=alpha)
        print('MobileNet:\n')
        #conv_base.summary()
        '''
        alpha: 控制网络的宽度：
                  如果alpha<1，则同比例的减少每层的滤波器个数
                  如果alpha>1，则同比例增加每层的滤波器个数
                  如果alpha=1，使用默认的滤波器个数
        '''
        model.add(conv_base)
        model.add(layers.Reshape((1, 1, int(1024 * alpha))))
        model.add(layers.Dropout(1e-3))  # see default parameter 'dropout=1e-3'in keras.applications.MobileNet() for more detail.
        model.add(layers.Conv2D(len(self.labels), (1, 1), padding='same', name='conv_preds'))
        model.add(layers.Activation('softmax', name='act_softmax'))
        model.add(layers.Reshape((len(self.labels),), name='reshape_2'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optim, metrics=['acc'])
        return model


    def SkirtVisableNet(self): # useless model.....just for test....
        model = models.Sequential()
        model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape, padding="same")) # 4*(64*3*3+64)
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same")) # 4*(128*4*4+128)
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same")) # 4*(128*4*4+128)
        model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=(3,3), padding="same"))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same")) # 4*(128*4*4+128)
        model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding="same"))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same")) # 4*(128*4*4+128)
        model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding="same"))
        model.add(layers.Flatten())
        model.add(layers.Dense(192, activation='relu'))# 4*( 32*32*128 +128)
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(80, activation='relu'))# 4*( 32*32*128 +128)
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optim, metrics=['acc'])
        # model.add(layers.Dense(1, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy',
        #                   optimizer=self.optim,
        #                   metrics=['acc'])
        return model

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