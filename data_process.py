from model_train_para import model_para
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import numpy as np


class Image_process(model_para):

    def _annotate_image(self):
        labels_distribution = self.df['labels'].value_counts()
        print('数据类别分布：\n', labels_distribution)
        labels = labels_distribution.index
        for i in range(len(labels_distribution)):
            label = labels[i]
            label_num = labels_distribution[i]
            image_path = self.df[self.df['labels'] == label]  # 这里最好用pd.groupby('labels')来处理
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

    def copy_image(self, images, path):
        for image in images:
            src = os.path.join(self.origin_dir, image)
            dst = os.path.join(path, image)
            with Image.open(src) as img:
                img = img.resize((self.input_shape[0], self.input_shape[1]), Image.ANTIALIAS)
                img.save(dst)

    def count_images(self, file_name):
        path = os.path.join(self.origin_dir, file_name)
        num = 0
        for label_file in os.listdir(path):
            label_path = os.path.join(path, label_file)
            num += len(os.listdir(label_path))
        return num

    def image_process(self):
        is_annotate_image = False
        for file in self.files:
            datafile = os.path.join(self.origin_dir, file)
            if not os.path.exists(datafile):
                os.makedirs(datafile)
                is_annotate_image = True
            else:
                is_annotate_image = False
                break  # 数据文件如果已经存在，就没必要继续对数据进行处理了
        if is_annotate_image:
            self._annotate_image()

    def random_crop(x, random_crop_size, sync_seed=None, **kwargs):
        np.random.seed(sync_seed)
        w, h = x.shape[0], x.shape[1]
        rangew = (w - random_crop_size[0]) // 2
        rangeh = (h - random_crop_size[1]) // 2
        offsetw = 0 if rangew == 0 else np.random.randint(rangew)
        offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
        return x[offsetw:offsetw + random_crop_size[0], offseth:offseth + random_crop_size[1], :]

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
                                         rotation_range=10,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         shear_range=0.1,
                                         zoom_range=0.1,
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
