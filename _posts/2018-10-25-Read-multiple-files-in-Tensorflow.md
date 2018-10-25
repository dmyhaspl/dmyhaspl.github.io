---
layout:     post                    # 使用的布局（不需要改）
title:      Read multiple files in Tensorflow               # 标题 
subtitle:    Read multiple sample files through the file queue #副标题
date:       2018-10-25             # 时间
author:     deepmyhaspl                     # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - AI
---


>Read multiple sample files through the file queue



```python

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 10:54:53 2018

@author: myhaspl
@email:myhaspl@myhaspl.com
 Read multiple sample files through the file queue

"""
import tensorflow as tf
import os

validateCount=10
sampleCount=10
testCount=10


g=tf.Graph()

with g.as_default():


    def inputFromFile(fileName,skipLines=1):
        #Generate file name queues
        fileNameQueue=tf.train.string_input_producer(fileName)
        #Generate record key pairs
        reader=tf.TextLineReader(skip_header_lines=skipLines)
        key,value=reader.read(fileNameQueue)
        return key,value


    
    with tf.name_scope("inputSample"): 
        mykey,mysamples=inputFromFile([os.getcwd()+"/1-1.csv",os.getcwd()+"/1-2.csv"],1)

  
        
 
with tf.Session(graph=g) as sess:
    # Start generating file name queues
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    num_examples = 0
    try:
        while True:
            key,value = sess.run([mykey, mysamples])
            print(key,value)
        num_examples += 1
    except tf.errors.OutOfRangeError:
        print ("There are", num_examples, "examples")


    

    coord.request_stop()
    coord.join(threads)

```

The program runs as follows:
```
(’1-1.csv:4’, ‘11,50,1’)
(1-1.csv:5’, ‘1.89,66,23’)
(’1-1.csv:6’, ‘94,28.1,0.167’)
(’1-1.csv:7’, ‘22,21,0.9’)
(’1-2.csv:2’, ‘11,21,10’)
(’1-2.csv:3’, ‘1,41,39’)
(’1-2.csv:4’, ‘10,32,4’)
(’1-2.csv:5’, ‘2.14,91.2,0.92’)
(’1-2.csv:6’, ‘109.3,19.03,81.02’)
(’1-2.csv:2’, ‘11,21,10’)
(’1-2.csv:3’, ‘1,41,39’)
(’1-2.csv:4’, ‘10,32,4’)
(’1-2.csv:5’, ‘2.14,91.2,0.92’)
(’1-2.csv:6’, ‘109.3,19.03,81.02’)

```

```
1-1.csv
x1,x2,y
6.148,72,35.88
0,33.6,0.627
11,50,1
1.89,66,23
94,28.1,0.167
22,21,0.9

1-2.csv
x1,x2,y
11,21,10
1,41,39
10,32,4
2.14,91.2,0.92
109.3,19.03,81.02
```