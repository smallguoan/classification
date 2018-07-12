import Classification_label as Cl
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import layers

class network:
    def __init__(self,batch_size,learning_rate,MAX_iteration):
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.MAX_iteration=MAX_iteration

        self.build_network()
        #self.sess=tf.Session()
        #self.sess.run(tf.global_variables_initializer())

    def build_network(self):
        # Define all the compentents for network
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
            return tf.Variable(initial)
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
            return tf.Variable(initial)
        def conv2d(x, W,stride):
            # stride[1,x_movement,y_movement,1]
            # Must have stride[0]=1 stride[3]=1
            return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        def aver_pool_2x2(x):
            return tf.nn.avg_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        # Define the input and the label for output
        self.input=tf.layers.Input(shape=[135,240,3],batch_size=self.batch_size,name='input',dtype=tf.float32)
        #self.input=tf.placeholder(tf.float16,shape=[None,270,480,3])
        self.output_label=tf.placeholder(tf.float32,shape=[None,2])

        #Define the network which is based on YOLOv1
        # Conv_layer1
        conv_1=tf.layers.conv2d(self.input,64,[7,7],(2,2),'SAME',name='conv_1',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv1=weight_variable([7,7,3,64])
        # b_conv1=bias_variable([64])
        # Conv_output_1=tf.nn.leaky_relu(conv2d(self.input,W_conv1,stride=2)+b_conv1,alpha=0.1)
        output_1_0=tf.layers.max_pooling2d(tf.nn.leaky_relu(conv_1,alpha=0.1),[2,2],[2,2],padding='SAME')
        output_1=tf.layers.batch_normalization(output_1_0)
        #output_1=max_pool_2x2(Conv_output_1)   #最终维度：68x120x64

        # Conv_layer2
        conv_2 = tf.layers.conv2d(output_1, 192, [3, 3], (1, 1), 'SAME', name='conv_2',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv2=weight_variable([3,3,64,192])
        # b_conv2=bias_variable([192])
        # Conv_output_2=tf.nn.leaky_relu(conv2d(output_1,W_conv2,stride=1)+b_conv2,alpha=0.1)
        output_2_0 = tf.layers.max_pooling2d(tf.nn.leaky_relu(conv_2,alpha=0.1), [2, 2], [2, 2], padding='SAME')
        output_2 = tf.layers.batch_normalization(output_2_0)
        #output_2=max_pool_2x2(Conv_output_2)    #最终维度：34x60x192

        # Conv_layer3
        conv_3=tf.layers.conv2d(output_2, 128, [1, 1], (1, 1), 'SAME', name='conv_3',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv3 = weight_variable([1, 1, 192, 128])
        # b_conv3 = bias_variable([128])
        output_3_0 = tf.nn.relu(conv_3)
        output_3 = tf.layers.batch_normalization(output_3_0)
        # 最终维度：34x60x128

        # Conv_layer4
        conv_4 = tf.layers.conv2d(output_3, 256, [3, 3], (1, 1), 'SAME', name='conv_4',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv4 = weight_variable([3, 3, 128, 256])
        # b_conv4 = bias_variable([256])
        output_4 = tf.nn.relu(conv_4)
        # 最终维度：34x60x256

        # Conv_layer5
        conv_5 = tf.layers.conv2d(output_4, 256, [1, 1], (1, 1), 'SAME',name='conv_5',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv5 = weight_variable([1, 1, 256, 256])
        # b_conv5 = bias_variable([256])
        output_5 = tf.nn.relu(conv_5)
        # 最终维度：34x60x256

        # Conv_layer6
        conv_6 = tf.layers.conv2d(output_5, 512, [3, 3], (1, 1), 'SAME', name='conv_6',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv6 = weight_variable([3, 3, 256, 512])
        # b_conv6 = bias_variable([512])
        # Conv_output_6 = tf.nn.leaky_relu(conv2d(output_5, W_conv6, stride=1) + b_conv6, alpha=0.1)
        output_6=tf.layers.max_pooling2d(tf.nn.relu(conv_6),[2,2],[2,2],padding='SAME')

        # output_6 = max_pool_2x2(Conv_output_6)
        # 最终维度：17x30x512

        # Conv layer7
        conv_7 = tf.layers.conv2d(output_6, 256, [1, 1], (1, 1), 'SAME',name='conv_7',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv7 = weight_variable([1, 1, 512, 256])
        # b_conv7 = bias_variable([256])
        output_7 = tf.nn.relu(conv_7)
        #最终维度：17x30x256

        # Conv layer8
        conv_8 = tf.layers.conv2d(output_7, 256, [1, 1], (1, 1), 'SAME', name='conv_8',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv8 = weight_variable([1, 1, 256, 256])
        # b_conv8 = bias_variable([256])
        output_8 = tf.nn.relu(conv_8)
        # 最终维度：17x30x256

        # Conv layer9
        conv_9 = tf.layers.conv2d(output_8, 256, [1, 1], (1, 1), 'SAME', name='conv_9',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv9 = weight_variable([1, 1, 256, 256])
        # b_conv9 = bias_variable([256])
        output_9 = tf.nn.relu(conv_9)
        # 最终维度：17x30x256

        # Conv layer10
        conv_10 = tf.layers.conv2d(output_9, 256, [1, 1], (1, 1), 'SAME', name='conv_10',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv10 = weight_variable([1, 1, 256, 256])
        # b_conv10 = bias_variable([256])
        output_10 = tf.nn.relu(conv_10)
        # 最终维度：17x30x256

        # Conv layer11
        conv_11=tf.layers.conv2d(output_10, 512, [3, 3], (1, 1), 'SAME', name='conv_11',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv11 = weight_variable([3, 3, 256, 512])
        # b_conv11 = bias_variable([512])
        output_11 = tf.nn.relu(conv_11)
        # 最终维度：17x30x512

        # Conv layer12
        conv_12 = tf.layers.conv2d(output_11, 512, [3, 3], (1, 1), 'SAME', name='conv_12',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv12 = weight_variable([3, 3, 512, 512])
        # b_conv12 = bias_variable([512])
        output_12 = tf.nn.relu(conv_12)
        # 最终维度：17x30x512

        # # Conv layer13
        # W_conv13 = weight_variable([3, 3, 512, 512])
        # b_conv13 = bias_variable([512])
        # output_13 = tf.nn.leaky_relu(conv2d(output_12, W_conv13, stride=1) + b_conv13, alpha=0.1)
        # # 最终维度：17x30x512
        #
        # # Conv layer14
        # W_conv14 = weight_variable([3, 3, 512, 512])
        # b_conv14 = bias_variable([512])
        # output_14 = tf.nn.leaky_relu(conv2d(output_13, W_conv14, stride=1) + b_conv14, alpha=0.1)
        # # 最终维度：17x30x512

        # Conv layer15
        conv_15 = tf.layers.conv2d(output_12, 512, [1, 1], (1, 1), 'SAME', name='conv_15',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv15 = weight_variable([1, 1, 512, 512])
        # b_conv15 = bias_variable([512])
        output_15 = tf.nn.relu(conv_15)
        # 最终维度：17x30x512

        # Conv layer16
        conv_16 = tf.layers.conv2d(output_15, 1024, [3, 3], (1, 1), 'SAME', name='conv_16',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv16 = weight_variable([3, 3, 512, 1024])
        # b_conv16 = bias_variable([1024])
        # Conv_output_16 = tf.nn.leaky_relu(conv2d(output_15, W_conv16, stride=1) + b_conv16, alpha=0.1)
        output_16=tf.layers.max_pooling2d(tf.nn.relu(conv_16),[2,2],[2,2],padding='SAME')
        #output_16=max_pool_2x2(Conv_output_16)
        # 最终维度：9x15x1024

        # Conv layer17
        conv_17 = tf.layers.conv2d(output_16, 512, [1, 1], (1, 1), 'SAME', name='conv_17',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv17 = weight_variable([1, 1, 1024, 512])
        # b_conv17 = bias_variable([512])
        output_17 = tf.nn.relu(conv_17)
        # 最终维度：9x15x512

        # Conv layer18
        conv_18 = tf.layers.conv2d(output_17, 512, [1, 1], (1, 1), 'SAME', name='conv_18',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv18 = weight_variable([1, 1, 512, 512])
        # b_conv18 = bias_variable([512])
        output_18 = tf.nn.relu(conv_18)
        # 最终维度：9x15x512

        # Conv layer19
        conv_19 = tf.layers.conv2d(output_18, 1024, [3, 3], (1, 1), 'SAME', name='conv_19',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv19 = weight_variable([3, 3, 512, 1024])
        # b_conv19 = bias_variable([1024])
        output_19 = tf.nn.relu(conv_19)
        # 最终维度：9x15x1024

        # Conv layer20
        conv_20 = tf.layers.conv2d(output_19, 1024, [3, 3], (1, 1), 'SAME', name='conv_20',kernel_regularizer=layers.l2_regularizer(0.1))
        # W_conv20 = weight_variable([3, 3, 1024, 1024])
        # b_conv20 = bias_variable([1024])
        output_20 = tf.nn.relu(conv_20)
        # 最终维度：9x15x1024

        # Average pooling layer and fully connected layer
        aver_output_0=tf.layers.average_pooling2d(output_20,[2,2],[2,2],padding='SAME')
        aver_output=tf.layers.batch_normalization(aver_output_0)
        #aver_output=aver_pool_2x2(output_20)
        #最终维度：5x8x1024
        W_fc1=weight_variable([3*4*1024,2])
        b_fc1=bias_variable([2])
        fc1_input=tf.reshape(aver_output,[-1,3*4*1024])
        self.result=tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(fc1_input,W_fc1)+b_fc1))
        self.loss=tf.reduce_mean(-tf.reduce_sum(self.output_label*tf.log(tf.clip_by_value(self.result,1e-10,1)),reduction_indices=[1])+layers.l2_regularizer(0.1)(W_fc1))
        self.train_step=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
        config.gpu_options.allow_growth = True
        self.sess=tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
    def train(self):
        # load the tfrecord
        coord = tf.train.Coordinator()
        image, label = Cl.read_and_decode('catch_classification_train.tfrecords', [135, 240, 3])
        label = tf.one_hot(label, 2, dtype=tf.float32)
        img_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=self.batch_size, capacity=1000,
                                                        min_after_dequeue=500, num_threads=1)
        image_eval, label_eval = Cl.read_and_decode('catch_classification_eval.tfrecords', [135, 240, 3])
        label_eval = tf.one_hot(label_eval, 2, dtype=tf.float32)
        img_eval_batch, label_eval_batch = tf.train.shuffle_batch([image_eval, label_eval],
                                                                  batch_size=self.batch_size, capacity=1000,
                                                                  min_after_dequeue=500, num_threads=1)
        threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)
        for i in range(self.MAX_iteration):


            val,l=self.sess.run([img_batch,label_batch])

            self.sess.run(self.train_step,feed_dict={self.input:val,self.output_label:l})
            print("this is the %d th "%i,"loss=",self.sess.run(self.loss,feed_dict={self.input:val,self.output_label:l}))
            if i%100==0:
                eval,l_eval=self.sess.run([img_eval_batch,label_eval_batch])
                print("eval=",self.accuracy(eval,l_eval))
                #coord.request_stop()
                #coord.join(threads)
        coord.request_stop()
        coord.join(threads)
        saver=tf.train.Saver()
        saver.save(self.sess,"/home/anguo/PycharmProjects/random_catch/pre_classification.ckpt")
    def accuracy(self,v_xs,v_ys):

        y_pre = self.sess.run(self.result, feed_dict={self.input: v_xs})  # ys:v_ys can be deleted
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = self.sess.run(accuracy, feed_dict={self.input: v_xs, self.output_label: v_ys})
        return acc
        
