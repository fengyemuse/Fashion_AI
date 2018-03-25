import os
from Model import Image_Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # windows使用这个命令
model_exist = False
model_class = Image_Model()
if not model_exist:
    model = model_class.create_model(base_model='MobileNet')
    history, true_sample, predicted_sample = model_class.train_model(model=model,
                                                                     is_augumente=True,
                                                                     is_image_processed=True)

    # 对图像进行处理后，记得把is_image_processed设置成False    (note: True)
    model_class.train_validation_result_plot(history, true_sample, predicted_sample)
else:
    model = model_class.model_load()
