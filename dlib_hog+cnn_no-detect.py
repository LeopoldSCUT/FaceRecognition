# -*- coding: UTF-8 -*-

import dlib,glob,numpy
from skimage import io
import pickle_example
import time
from itertools import islice

# 本文件先采取dlib库中的hog模型检测人脸是否存在，若不存在，再使用cnn模型检测，还不存在的话就忽略人脸检测

# 1.人脸关键点检测器

predictor_path = './shape_predictor_68_face_landmarks.dat'

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

# 3. 加载人脸识别模型

facerec = dlib.face_recognition_model_v1(face_rec_model_path)


# win = dlib.image_window()



# 候选人脸描述子list

#descriptors = []


# 对文件夹下的每一个人脸进行:


# 1.人脸检测


# 2.关键点检测


# 3.描述子提取

index_txt = open("./ValidationList.txt")
time_start = time.time()

count = count2 = 0

#for start in range(0,8):
# 每4314个样本计算一次，分批计算，start = 0 * 4314 to 7 * 4314
start = 6 * 4314
result = numpy.empty(shape=[0, 128], dtype='f')

for f in islice(index_txt, start + 2157, start + 4314):
    f = f.strip('\n')

    count += 1
    print("Processing {} file: {}".format(count, f))
    img = io.imread(f)

    # 1.人脸检测
    dets = detector(img, 1)

    if len(dets) == 0:
        dets2 = detector2(img, 1)
        if len(dets2) == 0:
            rec = dlib.rectangle(0, 0, img.shape[1], img.shape[0])
            shape = sp(img, rec)
            count2 += 1
            print("Number of faces detected: {}".format(len(dets2)))
        else:
            print("Number of faces detected by cnn: {}".format(len(dets2)))
            for k, d in enumerate(dets2):
                # 2.关键点检测
                # 画出人脸区域和和关键点
                rec = dlib.rectangle(d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
                shape = sp(img, rec)
    else:
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            # 2.关键点检测
            shape = sp(img, d)
            # 画出人脸区域和和关键点
            # 3.描述子提取，128D向量

    face_descriptor = facerec.compute_face_descriptor(img, shape)
    # 转换为numpy array
    v = numpy.array(face_descriptor, dtype='f')

    result = numpy.append(result, [v], axis=0)

pickle_example.save_feature(result, outputfile=str(count + start) + '.pkl')
print("Successfully store {}.pkl".format(count + start))

# pickle_example.save_feature(result, outputfile='version2.pkl')

time_end = time.time()
print('Total cost: ', time_end-time_start)
print("Failed result: {}".format(count2))
