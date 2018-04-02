import pandas as pd
import os
#some functions that may make you comfortable with your work.....


def generate_csv(path,subdirs,type_name,output_filename):
    """
    根据目录中的文件生成csv
    :param path:
    :param subdirs:
    :param type_name:
    :param output_filename:
    :return:
    """
    path_type=[(path,d) for d in subdirs]
    row1=[]
    row2=[]
    row3=[]
    for fd_path,label in path_type:
        for file in os.listdir(os.path.join(fd_path,label)):
            fullname = os.path.join(fd_path, label,file)
            row1.append(fullname)
            row2.append(type_name)
            row3.append(label)
    df=pd.DataFrame(
            {
                0:row1,
                1:row2,
                2:row3
            }
    )
    df.to_csv(output_filename,index=False,header=False)



if __name__ == '__main__':
    generate_csv(path=r"D:\Programming\fashionAI\data\warmup\Images\category64_expand",
                 subdirs=['true','false'],
                 type_name='skirt_visable',
                 output_filename='skirt_visable_label.csv'
                 )