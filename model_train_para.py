class model_para:
    def __init__(self):
        self.labels = ['/cats', '/dogs']
        self.input_shape = (150, 150, 3)
        self.label_types = 'binary'
        self.image_name = ['cat.{}.jpg', 'dog.{}.jpg']
        self.dirs = ['train', 'validation', 'test']
        self.files = [x + y for x in self.dirs for y in self.labels]
        self.data_split = [1000, 500, 500]  # 1000个训练，500个validate 500个test

        self.origin_dir = __file__.split(sep='dogs-vs-cats')[0] \
                     + 'data/dogs-vs-cats/train/'
        self.model_save_path = 'cats_and_dogs_small_1.h5'



















