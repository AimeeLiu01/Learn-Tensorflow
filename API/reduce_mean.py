# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:22:23 2018

@author: I332487
"""

import tensorflow as tf

x = tf.constant([[1., 2., 3.], [4., 5., 6.]])

with tf.Session() as sess:
    x = sess.run(x)
    #函数解说：將tensor取平均，第二個参数代表沿着那一維取平均， 
    #例如范例2，沿著第0維也就是 列取平均     第一维的元素取平均值，即每一列求平均值
    #再如范例3，沿著第1維也就是 行取平均     第二维的元素取平均值，即每一行求平均值
    mean1 = sess.run(tf.reduce_mean(x))
    mean2 = sess.run(tf.reduce_mean(x, 0))
    mean3 = sess.run(tf.reduce_mean(x, 1))
    
    print(x)
    print()
    print(mean1)
    print()
    print(mean2)
    print()
    print(mean3)
    print()
    

# [[1. 2. 3.]
# [4. 5. 6.]]

# 3.5

# [2.5 3.5 4.5]

# [2. 5.]
