# -*- coding: UTF-8 -*-

import dlib,glob,numpy,os
from skimage import io
import pickle_example
import time
from itertools import islice

# 1.人脸关键点检测器

predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor5_path = './shape_predictor_5_face_landmarks.dat'

# 2.人脸识别模型

face_rec_model_path = './dlib_face_recognition_resnet_model_v1.dat'

# 3.候选人脸文件夹

faces_folder_path = './ValidationData'

# 4.需识别的人脸

img_path = './ValidationData/00006.jpg'
img_path2 = './ValidationData/00009.jpg'

# 1.加载正脸检测器
# 传统的HOG特征+级联分类的方法
detector = dlib.get_frontal_face_detector()
# 使用卷积神经网络（CNN）进行人脸检测的检测算子
detector2 = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')


# 2.加载人脸关键点检测器

sp = dlib.shape_predictor(predictor_path)

sp5 = dlib.shape_predictor(predictor5_path)

# 3. 加载人脸识别模型

facerec = dlib.face_recognition_model_v1(face_rec_model_path)


# win = dlib.image_window()



# 候选人脸描述子list

#descriptors = []


# 对文件夹下的每一个人脸进行:


# 1.人脸检测


# 2.关键点检测


# 3.描述子提取

index_txt=open("./ValidationList.txt")
time_start=time.time()

count = 0
count2 = 0

#for start in range(0,8):
# 每4314个样本计算一次，分批计算，start = 0 * 4314 to 7 * 4314
# start = 1 * 4314
result = numpy.empty(shape=[0, 128], dtype='f')

# for f in islice(index_txt, start, start + 4314):
#     #print(start)
#     f = f.strip('\n')

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):

    count += 1
    print("Processing {} file: {}".format(count, f))
    img = io.imread(f)

    rec = dlib.rectangle(0, 0, img.shape[1], img.shape[0])

    shape = sp(img, rec)
    # 3.描述子提取，128D向量
    face_descriptor = facerec.compute_face_descriptor(img, shape)

    v = numpy.array(face_descriptor, dtype='f')
    # print("Feature:", v)

    result = numpy.append(result, [v], axis=0)

# pickle_example.save_feature(result, outputfile=str(count + start) + '.pkl')
# pickle_example.save_feature(result, outputfile='cnn-50-2.pkl')
# print(result.dtype)
# numpy.savetxt('cnn-50-2.txt', result)
# print("Successfully store {}.pkl".format(count + start))

pickle_example.save_feature(result, outputfile='version6-no-detect.pkl')

time_end=time.time()
print('Total cost: ',time_end-time_start)
# print("Failed result: {}".format(count2))
