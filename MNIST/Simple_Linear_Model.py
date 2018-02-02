# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:46:00 2018

@author: I332487
"""

### Imports 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


print(tf.__version__)

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

# 打印数据集的大小
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

print(data.test.labels[0:5, :])

img_size = 28
img_size_flat = img_size & img_size
img_shape = (img_size, img_size)
num_classes = 10

print(img_shape)
data.test.cls = np.array([label.argmax() for label in data.test.labels])
print(data.test.cls[0:5])

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)# 将画布分割成3行3列
    fig.subplots_adjust(hspace=0.3, wspace=0.3) 
    # 其中wspace hspace是用来控制宽度和高度百分比, 是用来控制横向和纵向subplot之间的间距
    
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
        


# -*- 绘制几张图像来看看数据是否正确 -*-
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images=images, cls_true=cls_true)


## ---------------------------------------------------------------------------
# 首先我们为输入图像定义placeholder变量 这让我们可以改变输入到Tensorflow图中的图像
# 这也是一个张量、代表一个多维向量或者矩阵
# 数据类型为float32  形状设为[None, img_size_flat]
# None 代表tensor可能保存着任意数量的图像 每张图像为一个长度为img_size_flat的向量
x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))
logits = tf.matmul(x, weights) + biases
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)

## 目标函数
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_true)
# 代价函数
cost = tf.reduce_mean(cross_entropy)
# 梯度下降法求代价函数的最小值 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
# This is a vector of booleans whether the predicted class equals the true class of each image.代表是否准确的预测  
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# This calculates the classification accuracy by first type-casting the vector of booleans to floats, 
# so that False becomes 0 and True becomes 1, and then calculating the average of these numbers.
accracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())


###-有在训练集50.000图像。 使用所有这些图像来计算模型的梯度需要很长时间。 
# 因此，我们使用随机梯度下降，它只在优化器的每次迭代中使用一小批图像。
batch_size = 100

def optimize(num_iterations):
    for i in range(num_iterations):
        # 获取一批测试数据
        # x_batch 代表一批图像  and y_true_batch 代表这批图像的值 .
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        
        # 将这批图像放入一个字典中 
        # 其中 y_true_cls不用放在集合中 因为在训练的过程中没有用到 
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

# Test-set data 被用作Tensorflow的输入
feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}
        

def print_accuracy():
    # Use TensorFlow to compute the accuracy.计算准确度
    acc = session.run(accracy, feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))
    

def print_confusion_matrix():
    # Get the true classifications for the test-set 得到测试数据集的真实类
    cls_true = data.test.cls
    # Get the predicted classifications for the test-set 得到测试数据集的预测类 
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred) # 得到混淆矩阵 
    
    # Print the confusion matrix as text.
    print(cm)
    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Make various adjustments to the plot
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    

def plot_example_errors():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified./得到被错误分类的图片 
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images. 得到这些错误分类图片的预测类 
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images. 得到这些错误分类图片的真实类 
    cls_true = data.test.cls[incorrect] 
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
    
    

def plot_weights():
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weights)
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# The accuracy on the test-set is 9.8%.
#  This is because the model has only been initialized and not optimized at all, so it always predicts that the image shows a zero digit,
print_accuracy()
plot_example_errors()

print("******************************************")
print("Performance after 1 optimization iteration")
optimize(num_iterations=1)
print_accuracy()
plot_example_errors()

plot_weights()


print("******************************************")
print("Performance after 10 optimization iterations")
optimize(num_iterations=9)
print_accuracy()
plot_example_errors()
plot_weights()

print("******************************************")
print("Performance after 1000 opimization iterations")
optimize(num_iterations=990)
print_accuracy()
plot_example_errors()
plot_weights()
print_confusion_matrix()



