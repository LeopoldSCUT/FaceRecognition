# -*- coding: UTF-8 -*-

import sys,os,dlib,glob,numpy
from skimage import io
import pickle_example
import time
import face_recognition as fr
from itertools import islice

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

index_txt=open("./ValidationList.txt")
time_start=time.time()

count = 0
count2 = 0

#for start in range(0,8):
start = 0
result = numpy.empty(shape=[0, 128], dtype=float)

for f in islice(index_txt, start, start + 4314):
    #print(start)
    f = f.strip('\n')
    if len(f) == 0:
        break;
#for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):

    count += 1
    print("Processing {} file: {}".format(count, f))
    img = io.imread(f)
    #win.clear_overlay()
    #win.set_image(img)

    # 1.人脸检测
    #dets = detector(img, 1)
    dets2 = detector2(img, 1)
    #print(type(dets))
    #print(type(dets2))
    print("Number of faces detected: {}".format(len(dets2)))

    if len(dets2) == 0:
        v = numpy.empty(128, dtype=float)
        count2 += 1
    else:
        for k, d in enumerate(dets2):
            # 2.关键点检测
            #shape = sp(img, d)
            # 画出人脸区域和和关键点
            # win.clear_overlay()
            # win.add_overlay(d)
            # win.add_overlay(shape)
            rec = dlib.rectangle(d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
            #print(rec.left(), rec.top(), rec.right(), rec.bottom())
            shape = sp(img, rec)

            # 3.描述子提取，128D向量
            face_descriptor = facerec.compute_face_descriptor(img, shape)

            # 转换为numpy array
            v = numpy.array(face_descriptor, dtype=float)
            #print("Feature:", v)

    result = numpy.append(result, [v], axis=0)

pickle_example.save_feature(result, outputfile=str(count) + '.pkl')
print("Successfully store {}.pkl".format(count))

#pickle_example.save_feature(result, outputfile='version3.pkl')

time_end=time.time()
print('Total cost: ',time_end-time_start)
print("Failed result: {}".format(count2))

'''


# 对需识别人脸进行同样处理


# 提取描述子，不再注释
x = numpy.empty(shape=[0, 128], dtype=float)
print(x.shape)
img = io.imread(img_path)
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))


if len(dets) == 0:
    d_test = numpy.zeros(128, dtype=float)
    x = numpy.append(x, [d_test], axis = 0)

for k, d in enumerate(dets):
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    d_test = numpy.array(face_descriptor, dtype=float)
    x = numpy.append(x, [d_test], axis = 0)

print("Feature:", d_test)

img = io.imread(img_path2)
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

if len(dets) == 0:
    d_test = numpy.zeros(128, dtype=float)
    x = numpy.append(x, [d_test], axis = 0)

for k, d in enumerate(dets):
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    d_test = numpy.array(face_descriptor, dtype=float)
    x = numpy.append(x, [d_test], axis = 0)

print("Feature:", d_test)
print("Result:", x)
print(x.shape)'''
'''
    # 计算欧式距离
    for i in descriptors:
        dist_ = numpy.linalg.norm(i-d_test)
        dist.append(dist_)


# 候选人名单

candidate = ['Unknown1','Unknown2','Shishi','Unknown4','Bingbing','Feifei']


# 候选人和距离组成一个dict

c_d = dict(zip(candidate,dist))

cd_sorted = sorted(c_d.iteritems(), key=lambda d:d[1])
print("\n The person is: " + cd_sorted[0][0])
dlib.hit_enter_to_continue()
'''
