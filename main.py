import os
from Model import Image_Model

"""
new feature needed:
1) (solved) confirm whether self.label is valid. i.e. I need not delete all the categories with m manually
2) (solved) model path should be saved after training: defined in 'model_train_para.py'
3) (solved) imaguag lib to augment images.
4）(solved) image mean and std normalization.
5) (solved) a file that describe the division of the the dataset. i.e a random seed --- memo part finished.. need to read from this file..
6) (solved) move optimizer configuration to 'model_train_para.py'
8) (solved) make sure csv in other cloth categories can be parsed correctly.
9) (solved) make correct prediction according to test.csv, and write corresponding results.
10) (solved) add a switch that could remove current data separation...
11) (solved) mapping between one-hot and self.labels
12）(solved) add augmentation to 'predict to csv'
13) (solved) tidy the parameters...
X) double-check details before final training...
"""



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # windows使用这个命令
model_class = Image_Model()
print(model_class.work_mode)
if model_class.work_mode == "train":
    print("entering train mode")
    model = model_class.create_model()
    history, true_sample, predicted_sample = model_class.train_model(model=model, is_augumente=True)
    model_class.train_validation_result_plot(history, true_sample, predicted_sample)
elif model_class.work_mode == "predict":
    print("entering test mode")
    model_class.predict_to_csv()
else:
    print("please choose a valid work mode!")

