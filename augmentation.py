import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
from model_train_para import model_para
import matplotlib.pyplot as plt


class augment(model_para):
    def __init__(self):
        model_para.__init__(self)

        ia.seed(1)
        # Example batch of images.

        """
        pay attention to speed when using augmentation....
        some transformation is extremely slow....
        I wonder whether this could be as fast as the intrinsic transformations of keras, so maybe we should only use it
        as a supplement of ImageDataGenerator


        # official tutorial and documentation:
        # http://imgaug.readthedocs.io/en/latest/
        # how to use it with keras: https://github.com/aleju/imgaug/issues/66

        """
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
        # image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image.
        self.seq = iaa.Sequential([
                iaa.Fliplr(0.5), # horizontal flips
                iaa.Crop(percent=(0, 0.01)), # random crops
                # Small gaussian blur with random sigma between 0 and 0.1.
                # But we only blur about 50% of all images.
                iaa.Sometimes(0.1,
                    iaa.GaussianBlur(sigma=(0, 0.1))
                ),
                # Strengthen or weaken the contrast in each image.
                iaa.ContrastNormalization((0.9, 1.1)),
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.01),
                # Make some images brighter and some darker.
                # In 20% of all cases, we sample the multiplier once per channel,
                # which can end up changing the color of the images.
                iaa.Multiply((0.95, 1.05), per_channel=0.1),
                # Apply affine transformations to each image.
                # Scale/zoom them, translate/move them, rotate them and shear them.
                iaa.Affine(
                    scale={"x": (0.95, 1.05), "y": (0.99, 1.01)},
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.002, 0.002)},
                    rotate=(-2, 2),
                    shear=(-2, 2)
                )
            ], random_order=True) # apply augmenters in random order


x_aug=augment()

#this function is called before any other transformations defined in ImageDataGenerator
def augment_callback(image):
    #print("augment!!!")
    images_aug = x_aug.seq.augment_image(image)
    #mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]),
    #but mean and std is considered in ImageDataGenerator...
    return images_aug.astype(np.float)


def augment_callback_predict(image):
    images_aug = x_aug.seq.augment_image(image)
    return images_aug.astype(np.float)

#for test only, to estimate the influences introduced to each image....
if __name__=="__main__":
    a=augment()
    columns = 10
    rows = 4
    with Image.open(r"D:\Programming\fashionAI_framework_v2\formal\Images\pant_length_labels\000032b036c7c230827ba9505ff9df32.jpg") as image:
        imgarr = np.array(image.convert("RGB"))
        img_list=[]
        print("start to augment images....")
        for i in range(columns*rows):
            x=augment_callback(imgarr).astype(np.uint8)
            img=Image.fromarray(x)
            #img=img.resize((224,224))
            img_list.append(img)
        print("start to plot....")
        fig=plt.figure()
        fig.subplots_adjust(hspace=0.01, wspace=0.01)

        for i in range(1,1+len(img_list)):
            fig.add_subplot(rows, columns, i)
            plt.axis('off')
            plt.imshow(img_list[i-1])

        plt.show()






















#####################################################################################
# import pandas as pd
# import os
# def output_to_csv(csv_pathname,dic_name_res):
#     df=pd.read_csv(csv_pathname, header=None)
#     df.columns = ('pathname', 'types', 'labels')
#
#     for filename,val in dic_name_res.items():
#         try:
#             df.loc[df['pathname']=="Images/"+df['types']+"/"+filename,'labels']=str(val)
#         except:
#             print("error!{}".format(filename,val))
#
#     df.to_csv(csv_pathname,index=False,header=False)


# d={"60951f5761ea3a01d6ed80e76b7f81ac.jpg":[121232131231231,2,3],"677e1183282769a3fe8ac5f1f0154bbd.jpg":[121232131231231,2,3],"1.ss":[22222222222222222222]}
# output_to_csv("question.csv",d)