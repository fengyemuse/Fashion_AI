from file_process import file_process
from model_train_para import model_para
from keras.preprocessing.image import ImageDataGenerator

import os


class image_process(model_para):

    def annotate_image(self):

        target_dict = {}
        for i in range(self.datalen):
            image = self.df.loc[i][0].split('/')[-1]
            target = self.df.loc[i][2]
            if target in target_dict.keys():
                target_dict[target] += 1
            else:
                target_dict[target] = 1
            target += '.' + str(target_dict[target]) + '.jpg'
            src = os.path.join(self.origin_dir, image)
            dst = os.path.join(self.origin_dir, target)
            if os.path.exists(src):
                os.rename(src=src, dst=dst)

    def image_cut_glue(self):
        '''
        主要是将图像剪切到指定的文件夹
        :param image_name: 图像的名字
        :param files: 拷贝的目标文件夹
        :param data_split: 数据集
        :return:
        '''

        data_file_manipulate = file_process(self.origin_dir)
        file_paths = dict()
        for file in self.files:
            file_paths[file] = data_file_manipulate.create_file(file)

        for i in range(len(self.image_name)):
            train_image = [self.image_name[i].format(j) for j in range(self.data_split[0])]
            data_file_manipulate.cut_image_to_file(train_image, file_paths[self.files[i]])
            validate_image = [self.image_name[i].format(j) for j in
                              range(self.data_split[0], self.data_split[0] + self.data_split[1])]
            data_file_manipulate.cut_image_to_file(validate_image, file_paths[self.files[i + len(self.image_name)]])
            test_image = [self.image_name[i].format(j) for j in
                          range(self.data_split[0] + self.data_split[1], sum(self.data_split))]
            data_file_manipulate.cut_image_to_file(test_image, file_paths[self.files[i + 2 * len(self.image_name)]])

    def image_dataGen(self, directory, target_size, batch_size, data_augmentation=False):
        '''
        产生batch
        :param directory: 'train','validate','test'文件夹路径
        :param batch_size:
        :param target_size: 图像大小
        :return:
        '''
        if data_augmentation:
            datagen = ImageDataGenerator(rescale=1.0 / 255,
                                         rotation_range=40,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True,
                                         fill_mode='nearest')
        else:
            datagen = ImageDataGenerator(rescale=1.0 / 255)
        data_generator = datagen.flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=self.label_types
        )
        return data_generator
