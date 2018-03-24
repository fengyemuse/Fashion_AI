from model_train_para import model_para
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import numpy as np


class image_process(model_para):

    def create_file(self, file_name):
        datafile = os.path.join(self.origin_dir, file_name)
        if not os.path.exists(datafile):
            os.makedirs(datafile)
        return datafile

    def copy_image(self, images, path):
        for image in images:
            src = os.path.join(self.origin_dir, image)
            dst = os.path.join(path, image)
            with Image.open(src) as img:
                img = img.resize((self.input_shape[0], self.input_shape[1]), Image.ANTIALIAS)
                img.save(dst)

    def annotate_image(self):

        file_paths = dict()
        for file in self.files:
            file_paths[file] = self.create_file(file)
        labels_distribution = self.df['labels'].value_counts()
        print('数据类别分布：\n', labels_distribution)
        labels = labels_distribution.index
        for i in range(len(labels_distribution)):
            label = labels[i]
            label_num = labels_distribution[i]
            image_path = self.df[self.df['labels'] == label]   # 这里最好用pd.groupby('labels')来处理
            '''for name,group in df.groupby('labels'):  
                    print(name)  
                    print(group) 
                     '''
            image_names = np.array([image.split('/')[-1] for image in image_path['picture']])
            np.random.shuffle(image_names)  # 打乱数据
            train_path = os.path.join(self.origin_dir, self.files[i])
            validate_path = os.path.join(self.origin_dir, self.files[i + len(labels_distribution)])
            test_path = os.path.join(self.origin_dir, self.files[i + 2 * len(labels_distribution)])
            if label_num < 3:  # 即无法分开数据
                for name in image_names:
                    src = os.path.join(self.origin_dir, name)
                    if os.path.exists(src):
                        with Image.open(src) as img:
                            img = img.resize((self.input_shape[0], self.input_shape[1]), Image.ANTIALIAS)
                            img.save(os.path.join(train_path, name))
                            img.save(os.path.join(validate_path, name))
                            img.save(os.path.join(test_path, name))
            else:
                test_image_num = max(round(self.data_split_ratio[2] * image_names.size), 1)
                validate_image_num = max(round(self.data_split_ratio[1] * image_names.size), 1)

                test_image = image_names[:test_image_num]
                validate_image = image_names[test_image_num:test_image_num + validate_image_num]
                train_image = image_names[test_image_num + validate_image_num:]
                self.copy_image(test_image, test_path)
                self.copy_image(validate_image, validate_path)
                self.copy_image(train_image, train_path)

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
