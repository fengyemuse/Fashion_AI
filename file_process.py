import os


class file_process:
    def __init__(self, origin_dir):

        '''
        :param origin_dir: 数据存放的文件路径
        '''

        self.origin_dir = origin_dir

    def create_file(self, file_name):
        datafile = os.path.join(self.origin_dir, file_name)
        if not os.path.exists(datafile):
            os.makedirs(datafile)
        return datafile

    def cut_image_to_file(self, images, dir):
        '''

        :param images: 图片名
        :param dir: 拷贝到的文件夹路径
        '''
        for image in images:
            src = os.path.join(self.origin_dir, image)
            dst = os.path.join(dir, image)
            if os.path.exists(src):
                os.rename(src, dst)  # 图片的剪切复制
