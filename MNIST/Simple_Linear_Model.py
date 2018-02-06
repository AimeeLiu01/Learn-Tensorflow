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
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

# 打印数据集的大小
# 现在已经载入了MNIST数据集，它由70,000张图像和对应的标签（比如图像的类别）组成。
# 数据集分成三份互相独立的子集。我们在教程中只用训练集和测试集。
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

# One-Hot 编码。数据集以一种称为One-Hot编码的方式载入。
# 这意味着标签从一个单独的数字转换成一个长度等于所有可能类别数量的向量。
# 向量中除了第$i$个元素是1，其他元素都是0，这代表着它的类别是$i$'。
# 比如，前面五张图像标签的One-Hot编码为：
print(data.test.labels[0:5, :])

img_size = 28
img_size_flat = img_size & img_size
img_shape = (img_size, img_size)
num_classes = 10

print(img_shape)
# 我们也需要用单独的数字表示类别，因此我们通过取最大元素的索引，
# 将One-Hot编码的向量转换成一个单独的数字。需注意的是'class'在Python中是一个关键字，所以我们用'cls'代替它。
data.test.cls = np.array([label.argmax() for label in data.test.labels])
print(data.test.cls[0:5])

# 数据维度
# 在下面的源码中，有很多地方用到了数据维度。在计算机编程中，通常来说最好使用变量和常量，而不是在每次使用数值时写硬代码。这意味着数字只需要在一个地方改动就行。这些最好能从读取的数据中获取，但这里我们直接写上数值。

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28
# Images are stored in one-dimensional arrays of this length. 每一个图片都被存储在28*28的一维向量中
img_size_flat = img_size * img_size
# 用于重塑数组的图像高度和宽度的元组。
img_shape = (img_size, img_size)
# Number of classes, one class for each of 10 digits.
num_classes = 10

## 这个函数用来在3x3的栅格中画9张图像，然后在每张图像下面写出真实的和预测的类别。
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
plot_images(images=images, cls_true=cls_true) # 调用绘图函数检查正确性 

#-----------------------------------------------------------------------------#
# TensorFlow的全部目的就是使用一个称之为计算图（computational graph）的东西，
# 它会比直接在Python中进行相同计算量要高效得多。TensorFlow比Numpy更高效，因为TensorFlow了解整个需要运行的计算图，
# 然而Numpy只知道某个时间点上唯一的数学运算。

# 一个TensorFlow图由下面详细描述的几个部分组成：

# 占位符变量（Placeholder）用来改变图的输入。
# 模型变量（Model）将会被优化，使得模型表现得更好。
# 模型本质上就是一些数学函数，它根据Placeholder和模型的输入变量来计算一些输出。
# 一个cost度量用来指导变量的优化。
# 一个优化策略会更新模型的变量。
#-----------------------------------------------------------------------------#

# 这也是一个张量（tensor），代表一个多维向量或矩阵。数据类型设置为float32，
# 形状设为[None, img_size_flat]，None代表tensor可能保存着任意数量的图像，每张图象是一个长度为img_size_flat的向量。
x = tf.placeholder(tf.float32, [None, img_size_flat])

# 输入变量x中的图像所对应的真实标签定义placeholder变量 变量的形状是[None, num_classes]，
# 这代表着它保存了任意数量的标签，每个标签是长度为num_classes的向量，本例中长度为10。
y_true = tf.placeholder(tf.float32, [None, num_classes])

# 我们为变量x中图像的真实类别定义placeholder变量。 
y_true_cls = tf.placeholder(tf.int64, [None])

# 第一个需要优化的变量称为权重weight，TensorFlow变量需要被初始化为零，
# 它的形状是[img_size_flat, num_classes]，因此它是一个img_size_flat行、num_classes列的二维张量（或矩阵）。
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))

# 第二个需要优化的是偏差变量biases，它被定义成一个长度为num_classes的1维张量（或向量）。
biases = tf.Variable(tf.zeros([num_classes]))

# 这个最基本的数学模型将placeholder变量x中的图像与权重weight相乘，然后加上偏差biases。
# 由于x的形状是[num_images, img_size_flat] 并且 weights的形状是[img_size_flat, num_classes]，因此两个矩阵乘积的形状是[num_images, num_classes]
# 现在logits是一个 num_images 行num_classes列的矩阵，第$i$行第$j$列的那个元素代表着第 i 张输入图像有多大可能性是第 j 个类别。
logits = tf.matmul(x, weights) + biases

# 因此我们想要对它们做归一化，使得logits矩阵的每一行相加为1，每个元素限制在0到1之间。这是用一个称为softmax的函数来计算的，结果保存在y_pred中。
y_pred = tf.nn.softmax(logits)
# 可以从y_pred矩阵中取每行最大元素的索引值，来得到预测的类别。
y_pred_cls = tf.argmax(y_pred, axis=1)

#-----------------插入解释----------------------------------------------------#
# 为了使模型更好地对输入图像进行分类，我们必须改变weights和biases变量。
# 首先我们需要比较模型的预测输出y_pred和期望输出y_true，来了解目前模型的性能如何。
# 交叉熵（cross-entropy）是一个在分类中使用的性能度量。交叉熵是一个常为正值的连续函数，如果模型的预测值精准地符合期望的输出，它就等于零。
# 因此，优化的目的就是最小化交叉熵，通过改变模型中weights和biases的值，使交叉熵越接近零越好。

# 计算交叉熵的函数
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_true)

# 现在，我们已经为每个图像分类计算了交叉熵，所以有一个当前模型在每张图上的性能度量。
# 但是为了用交叉熵来指导模型变量的优化，我们需要一个额外的标量值，因此我们简单地利用所有图像分类交叉熵的均值。
cost = tf.reduce_mean(cross_entropy)


# 优化方法
# 梯度下降法求代价函数的最小值 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
# 代表是否准确的预测  结果为一个Bool值 True|False
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# 计算这些值的平均数，以此来计算分类的准确度。
accracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建TensorFlow会话（session）
session = tf.Session()
session.run(tf.global_variables_initializer())


# 有在训练集50.000图像。 使用所有这些图像来计算模型的梯度需要很长时间。 
# 因此，我们使用随机梯度下降，它只在优化器的每次迭代中使用一小批图像。
batch_size = 100

# 函数执行了多次的优化迭代来逐步地提升模型的weights和biases。
# 在每次迭代中，从训练集中选择一批新的数据，然后TensorFlow用这些训练样本来执行优化器。
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
        
# 用来打印测试集分类准确度的函数 
def print_accuracy():
    # Use TensorFlow to compute the accuracy.计算准确度
    acc = session.run(accracy, feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))
    

def print_confusion_matrix():
    # 得到测试数据集的真实类
    cls_true = data.test.cls
    # 得到测试数据集的预测类 
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
    

# 绘制测试集中误分类图像的函数。 
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
    
    

# 绘制模型权重的帮助函数
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



