from model_train_para import model_para
from augmentation import augment_callback
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import time,sys,os,pickle
import shutil
class Image_process(model_para):

    def _annotate_image(self):
        memo_test,memo_train,memo_valid=[],[],[]
        labels_distribution = self.df['labels'].value_counts()
        print('数据类别分布：\n', labels_distribution)
        labels = labels_distribution.index
        print("start annotating images.....")
        if self.data_sep_according_to_file:
                print("restoring annotation from memo.....")
                with open(self.sep_file_path,"rb") as f:
                    memo=pickle.load(f)

                created=set()#accelerate my kuso code...

                for target, fname in memo.items():
                    with Image.open(os.path.join(self.origin_dir,fname)) as img:
                        img = img.resize((self.input_shape[0], self.input_shape[1]), Image.ANTIALIAS)
                        p=os.path.split(target)[0]
                        if not os.path.exists(p) and p not in created:
                            os.makedirs(p)
                            created.add(p)
                        img.save(os.path.join(self.origin_dir,target))
                return


        memo={}
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
                            memo[os.path.join(self.files[i],name)]=name
                            memo[os.path.join(self.files[i+ len(labels_distribution)],name)]=name
                            memo[os.path.join(self.files[i+ 2*len(labels_distribution)],name)]=name
            else:
                test_image_num = max(round(self.data_split_ratio[2] * image_names.size), 1)
                validate_image_num = max(round(self.data_split_ratio[1] * image_names.size), 1)

                test_image = image_names[:test_image_num]
                validate_image = image_names[test_image_num:test_image_num + validate_image_num]
                train_image = image_names[test_image_num + validate_image_num:]
                self.copy_image(test_image, test_path)
                self.copy_image(validate_image, validate_path)
                self.copy_image(train_image, train_path)
                for name in train_image:
                    memo[os.path.join(self.files[i],name)]=name
                for name in validate_image:
                    memo[os.path.join(self.files[i + len(labels_distribution)],name)]=name
                for name in test_image:
                    memo[os.path.join(self.files[i + 2*len(labels_distribution)],name)]=name

        #this memo is not important, but it can be used as a backup in case of emergencies i.e. data confusion, data lost, etc.

        if not os.path.exists("memo"):
            os.makedirs("memo")

        filename="separate_memo-{}.pkl".format(str(time.strftime("%Y%m%d%H%M%S",time.localtime())))
        with open(os.path.join(os.path.split(__file__)[0],"memo",filename),"wb") as f:
            pickle.dump(memo,f)
        print("images annotated.....")


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
        if self.re_annotate:
            for x in self.dirs:
                path=os.path.join(self.origin_dir,x)
                if os.path.exists(path):
                    shutil.rmtree(path)

        for file in self.files:
            datafile = os.path.join(self.origin_dir, file)
            if not os.path.exists(datafile):
                os.makedirs(datafile)
                is_annotate_image = True
            else:
                if self.re_annotate:
                    is_annotate_image=True
                break  # 数据文件如果已经存在，就没必要继续对数据进行处理了

        if is_annotate_image:
            self._annotate_image()

    def image_dataGen(self, directory, target_size, batch_size, data_augmentation=False):
        '''
        产生batch
        :param directory: 'train','validate','test'文件夹路径
        :param batch_size:
        :param target_size: 图像大小
        :return:
        '''

        if data_augmentation:
            # use imgaug instead
            datagen = ImageDataGenerator(rescale=1.0 / 255,
                                         # rotation_range=10,
                                         # width_shift_range=0.2,
                                         # height_shift_range=0.2,
                                         # shear_range=0.1,
                                         # zoom_range=0.1,
                                         # horizontal_flip=True,
                                         fill_mode='nearest',
                                         preprocessing_function=augment_callback,
                                         featurewise_std_normalization=True,
                                         featurewise_center=True
                                         )
            # rescaling is before normalization
            datagen.mean=np.array([0.485, 0.456, 0.406])
            datagen.std=np.array([0.229, 0.224, 0.225])
        else:
            datagen = ImageDataGenerator(rescale=1.0 / 255,
                                         featurewise_std_normalization=True,
                                         featurewise_center=True
                                         )
            datagen.mean=np.array([0.485, 0.456, 0.406])
            datagen.std=np.array([0.229, 0.224, 0.225])
        data_generator = datagen.flow_from_directory(
            directory,
            classes=self.labels,# in order to mark self.labels[i] as one-hot(i), instead of a random combination
            target_size=target_size,
            batch_size=batch_size,
            class_mode=self.label_types
        )
        return data_generator



