import Classification_label as Cl
import tensorflow as tf
import numpy as np

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
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
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
        self.input=tf.placeholder(tf.float32,shape=[None,540,960,3])
        self.output_label=tf.placeholder(tf.float32,shape=[None,2])

        #Define the network which is based on YOLOv1
        # Conv_layer1
        W_conv1=weight_variable([7,7,3,64])
        b_conv1=bias_variable([64])
        Conv_output_1=tf.nn.leaky_relu(conv2d(self.input,W_conv1,stride=2)+b_conv1,alpha=0.1)
        output_1=max_pool_2x2(Conv_output_1)   #最终维度：135x240x64

        # Conv_layer2
        W_conv2=weight_variable([3,3,64,192])
        b_conv2=bias_variable([192])
        Conv_output_2=tf.nn.leaky_relu(conv2d(output_1,W_conv2,stride=1)+b_conv2,alpha=0.1)
        output_2=max_pool_2x2(Conv_output_2)    #最终维度：68x120x192

        # Conv_layer3
        W_conv3 = weight_variable([1, 1, 192, 128])
        b_conv3 = bias_variable([128])
        output_3 = tf.nn.leaky_relu(conv2d(output_2, W_conv3, stride=1) + b_conv3, alpha=0.1)
        # 最终维度：68x120x128

        # Conv_layer4
        W_conv4 = weight_variable([3, 3, 128, 256])
        b_conv4 = bias_variable([256])
        output_4 = tf.nn.leaky_relu(conv2d(output_3, W_conv4, stride=1) + b_conv4, alpha=0.1)
        # 最终维度：68x120x256

        # Conv_layer5
        W_conv5 = weight_variable([1, 1, 256, 256])
        b_conv5 = bias_variable([256])
        output_5 = tf.nn.leaky_relu(conv2d(output_4, W_conv5, stride=1) + b_conv5, alpha=0.1)
        # 最终维度：68x120x256

        # Conv_layer6
        W_conv6 = weight_variable([3, 3, 256, 512])
        b_conv6 = bias_variable([512])
        Conv_output_6 = tf.nn.leaky_relu(conv2d(output_5, W_conv6, stride=1) + b_conv6, alpha=0.1)
        output_6 = max_pool_2x2(Conv_output_6)
        # 最终维度：34x60x512

        # Conv layer7
        W_conv7 = weight_variable([1, 1, 512, 256])
        b_conv7 = bias_variable([256])
        output_7 = tf.nn.leaky_relu(conv2d(output_6, W_conv7, stride=1) + b_conv7, alpha=0.1)
        #最终维度：34x60x256

        # Conv layer8
        W_conv8 = weight_variable([1, 1, 256, 256])
        b_conv8 = bias_variable([256])
        output_8 = tf.nn.leaky_relu(conv2d(output_7, W_conv8, stride=1) + b_conv8, alpha=0.1)
        # 最终维度：34x60x256

        # Conv layer9
        W_conv9 = weight_variable([1, 1, 256, 256])
        b_conv9 = bias_variable([256])
        output_9 = tf.nn.leaky_relu(conv2d(output_8, W_conv9, stride=1) + b_conv9, alpha=0.1)
        # 最终维度：34x60x256

        # Conv layer10
        W_conv10 = weight_variable([1, 1, 256, 256])
        b_conv10 = bias_variable([256])
        output_10 = tf.nn.leaky_relu(conv2d(output_9, W_conv10, stride=1) + b_conv10, alpha=0.1)
        # 最终维度：34x60x256

        # Conv layer11
        W_conv11 = weight_variable([3, 3, 256, 512])
        b_conv11 = bias_variable([512])
        output_11 = tf.nn.leaky_relu(conv2d(output_10, W_conv11, stride=1) + b_conv11, alpha=0.1)
        # 最终维度：34x60x512

        # Conv layer12
        W_conv12 = weight_variable([3, 3, 512, 512])
        b_conv12 = bias_variable([512])
        output_12 = tf.nn.leaky_relu(conv2d(output_11, W_conv12, stride=1) + b_conv12, alpha=0.1)
        # 最终维度：34x60x512

        # Conv layer13
        W_conv13 = weight_variable([3, 3, 512, 512])
        b_conv13 = bias_variable([512])
        output_13 = tf.nn.leaky_relu(conv2d(output_12, W_conv13, stride=1) + b_conv13, alpha=0.1)
        # 最终维度：34x60x512

        # Conv layer14
        W_conv14 = weight_variable([3, 3, 512, 512])
        b_conv14 = bias_variable([512])
        output_14 = tf.nn.leaky_relu(conv2d(output_13, W_conv14, stride=1) + b_conv14, alpha=0.1)
        # 最终维度：34x60x512

        # Conv layer15
        W_conv15 = weight_variable([1, 1, 512, 512])
        b_conv15 = bias_variable([512])
        output_15 = tf.nn.leaky_relu(conv2d(output_14, W_conv15, stride=1) + b_conv15, alpha=0.1)
        # 最终维度：34x60x512

        # Conv layer16
        W_conv16 = weight_variable([3, 3, 512, 1024])
        b_conv16 = bias_variable([1024])
        Conv_output_16 = tf.nn.leaky_relu(conv2d(output_15, W_conv16, stride=1) + b_conv16, alpha=0.1)
        output_16=max_pool_2x2(Conv_output_16)
        # 最终维度：17x30x1024

        # Conv layer17
        W_conv17 = weight_variable([1, 1, 1024, 512])
        b_conv17 = bias_variable([512])
        output_17 = tf.nn.leaky_relu(conv2d(output_16, W_conv17, stride=1) + b_conv17, alpha=0.1)
        # 最终维度：17x30x512

        # Conv layer18
        W_conv18 = weight_variable([1, 1, 512, 512])
        b_conv18 = bias_variable([512])
        output_18 = tf.nn.leaky_relu(conv2d(output_17, W_conv18, stride=1) + b_conv18, alpha=0.1)
        # 最终维度：17x30x512

        # Conv layer19
        W_conv19 = weight_variable([3, 3, 512, 1024])
        b_conv19 = bias_variable([1024])
        output_19 = tf.nn.leaky_relu(conv2d(output_18, W_conv19, stride=1) + b_conv19, alpha=0.1)
        # 最终维度：17x30x1024

        # Conv layer20
        W_conv20 = weight_variable([3, 3, 1024, 1024])
        b_conv20 = bias_variable([1024])
        output_20 = tf.nn.leaky_relu(conv2d(output_19, W_conv20, stride=1) + b_conv20, alpha=0.1)
        # 最终维度：17x30x1024

        # Average pooling layer and fully connected layer
        aver_output=aver_pool_2x2(output_20)
        #最终维度：9x15x1024
        W_fc1=weight_variable([9*15*1024,2])
        b_fc1=bias_variable([2])
        fc1_input=tf.reshape(aver_output,[-1,9*15*1024])
        fc1_output=tf.nn.softmax(tf.matmul(fc1_input,W_fc1)+b_fc1)
        self.result=fc1_output
        self.loss=tf.reduce_mean(-tf.reduce_sum(self.output_label*tf.log(tf.clip_by_value(self.result,1e-10,1.0)),reduction_indices=[1]))
        self.train_step=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
    def train(self):
        # load the tfrecord
        image,label=Cl.read_and_decode('catch_classification_train.tfrecords',[540,960,3])
        label=tf.one_hot(label,2,dtype=tf.float32)
        img_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=self.batch_size, capacity=2000,
                                                        min_after_dequeue=1000)
        threads = tf.train.start_queue_runners(sess=self.sess)
        for i in range(self.MAX_iteration):
            val,l=self.sess.run([img_batch,label_batch])
            self.sess.run(self.train_step,feed_dict={self.input:val,self.output_label:l})
            if i%50==0:
                image_eval,label_eval=Cl.read_and_decode('catch_classification_eval.tfrecords',[540,960,3])
                label_eval=tf.one_hot(label_eval,2,dtype=tf.float32)
                img_eval_batch,label_eval_batch=tf.train.shuffle_batch([image_eval, label_eval],
                                                    batch_size=self.batch_size, capacity=2000,
                                                    min_after_dequeue=1000)
                eval,l_eval=self.sess.run([img_eval_batch,label_eval_batch])
                print(self.accuracy(eval,l_eval))

    def accuracy(self,v_xs,v_ys):

        y_pre = self.sess.run(self.result, feed_dict={self.input: v_xs})  # ys:v_ys can be deleted
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = self.sess.run(accuracy, feed_dict={self.input: v_xs, self.output_label: v_ys})
        return acc
        
