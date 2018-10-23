---
layout:     post                    # 使用的布局（不需要改）
title:      Extract image edges through Tensorflow               # 标题 
subtitle:   Convolution and pooling  #副标题
date:       2018-10-20              # 时间
author:     deepmyhaspl                     # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - AI
---


>Extracting the edges of the image through conv2d


The right side of the image below is the result of the extraction edge
![The result of a blurred image](https://dmyhaspl.github.io/postimages/2018102301-01.png)

```python

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:23:27 2018

@author: myhaspl
@email:myhaspl@myhaspl.com
tf.nn.conv2d+tf.nn.maxpool

Extract the edges first, then blur and dry
先提取边缘，再模糊去燥
"""

import tensorflow as tf
from PIL import Image    
import numpy as np




g=tf.Graph()

with g.as_default():

    def getImageData(fileNameList):
        imageData=[]
        for fn in fileNameList:        
            testImage = Image.open(fn).convert('L')   
            testImage.show() 
            imageData.append(np.array(testImage)[:,:,None])
        return np.array(imageData,dtype=np.float32)

    imageFn=("tractor.png",)
    imageData=getImageData(imageFn)
    testData=tf.constant(imageData)
    kernel=tf.constant(np.array(
            [
                   [[[0.]],[[1.]],[[0.]]],
                   [[[1.]],[[-4.]],[[1.]]], 
                   [[[0.]],[[1.]],[[0.]]]
            ])
            ,dtype=tf.float32)#3*3*1*1
    blurkernel=tf.constant(np.array(
            [
                   [[[1.]],[[1.]],[[1.]]],
                   [[[1.]],[[1.]],[[1.]]], 
                   [[[1.]],[[1.]],[[1.]]]
            ])/9.
            ,dtype=tf.float32)#3*3*1*1
    convData1=tf.nn.conv2d(testData,kernel,strides=[1,1,1,1],padding="SAME")
    poolData1=tf.nn.max_pool(convData1,ksize=[1,2,2,1],strides=[1,1,1,1],padding='VALID')
    convData2=tf.nn.conv2d(poolData1,blurkernel,strides=[1,1,1,1],padding="SAME")
    poolData2=tf.nn.avg_pool(convData2,ksize=[1,6,6,1],strides=[1,1,1,1],padding='VALID')

    y1=tf.cast(poolData1, dtype=tf.int32)
    y2=tf.cast(poolData2, dtype=tf.int32)
    init_op = tf.global_variables_initializer()
with tf.Session(graph=g) as sess:
    print testData.get_shape()
    print kernel.get_shape()
    resultData1=sess.run(y1)[0]
    resultData2=sess.run(y2)[0]
    resultData1=resultData1.reshape(resultData1.shape[0],resultData1.shape[1])
    resulImage1=Image.fromarray(255-np.uint8(resultData1),mode='L')   
    resulImage1.show()
    resultData2=resultData2.reshape(resultData2.shape[0],resultData2.shape[1])
    resulImage2=Image.fromarray(255-np.uint8(resultData2),mode='L')   
    resulImage2.show()
    print y1.get_shape()

```
All rights reserved
