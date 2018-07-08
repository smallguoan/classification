import tensorflow as tf
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def creat_record(file_path,classes,shape):
    # file_path:样本路径
    # classes：要分类的类别,类型为dict
    # shape:在打包成tfrecord前图片需要resize的大小,类型为tuple,格式为(长,宽)
    writer = tf.python_io.TFRecordWriter("catch_classification_eval.tfrecords")

    for index, name in enumerate(classes):
        class_path = file_path + name + '/'
        print(index)
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name  # 每一个图片的地址

            img = Image.open(img_path)
            img = img.resize(shape)
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()

def read_and_decode(filename,shape): # 读入tfrecords
    #shape:tensorflow格式的shape[高，长，通道数],类型：list
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, shape=shape)  #reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    return img, label

def eval_tfrecords(output_path,tfrecords_paths,samples_number,shape):
    # output_path:表示将从tfrecord里面生成的图片保存的位置
    # tfrecords_paths:表示tfrecord所在位置
    # samples_number：表示样本总数
    # shape: tensorflow格式的输出图片大小,[高,长,通道数]
    filename_queue = tf.train.string_input_producer([tfrecords_paths]) #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, shape=shape)#长和高对调！
    label = tf.cast(features['label'], tf.int32)
    with tf.Session() as sess: #开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        for i in range(samples_number):
            example, l = sess.run([image,label])#在会话中取出image和label
            img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
            img.save(output_path+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
            #print(example, l)
        coord.request_stop()
        coord.join(threads)

file_path='/home/anguo/evaluation/'
classes={'Uncatchable','Catchable'}  #label中0是不可抓 1为可抓
total_sample_numbers=len(os.listdir(file_path+'Catchable'))+len(os.listdir(file_path+'Uncatchable'))

# writer=tf.python_io.TFRecordWriter("catch_classification_train.tfrecords")
#
# for index,name in enumerate(classes):
#     class_path=file_path+name+'/'
#     for img_name in os.listdir(class_path):
#         img_path=class_path+img_name #每一个图片的地址
#
#         img=Image.open(img_path)
#         img= img.resize((960,540))
#         img_raw=img.tobytes()#将图片转化为二进制格式
#         example = tf.train.Example(features=tf.train.Features(feature={
#             "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#             'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#         })) #example对象对label和image数据进行封装
#         writer.write(example.SerializeToString())  #序列化为字符串
#
# writer.close()


#creat_record(file_path,classes,(480,270))
eval_tfrecords(file_path+'output/',"catch_classification_eval.tfrecords",total_sample_numbers,[270,480,3])


