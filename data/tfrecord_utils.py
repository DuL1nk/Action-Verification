import tensorflow as tf


def get_example_nums(tf_records_filenames):
    '''
    统计tf_records图像的个数(example)个数
    :param tf_records_filenames: tf_records文件路径
    :return:
    '''
    nums = 0
    for record in tf.python_io.tf_record_iterator(tf_records_filenames):
        nums += 1
    return nums

def write_tfrecord(input, output):
    # 借助于TFRecordWriter 才能将信息写入TFRecord 文件
    writer = tf.python_io.TFRecordWriter(output)

    # 读取图片并进行解码
    image = tf.read_file(input)
    image = tf.image.decode_jpeg(image)

    with tf.Session() as sess:
        image = sess.run(image)
        shape = image.shape
        # 将图片转换成string
        image_data = image.tostring()
        # 创建Example对象，并将Feature一一对应填充进去
        example = tf.train.Example(features=tf.train.Features(feature={
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data]))
        }
        ))
        # 将example序列化成string 类型，然后写入。
        writer.write(example.SerializeToString())
    writer.close()


def read_record(filename):
    '''
    解析record文件:源文件的图像数据是RGB,uint8,[0,255],一般作为训练数据时,需要归一化到[0,1]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param type:选择图像数据的返回类型
         None:默认将uint8-[0,255]转为float32-[0,255]
         normalization:归一化float32-[0,1]
         centralization:归一化float32-[0,1],再减均值中心化
    :return:
    '''
    # 创建文件队列，不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # 为文件队列创建一个阅读区
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)

    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'data': tf.FixedLenFeature([], tf.string),
        }
    )
    # 获得图像原始的数据
    # pdb.set_trace()
    tf_image = tf.decode_raw(features['data'], tf.uint8)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        image = sess.run(tf_image)
        image = image.reshape([180, 320, 3])

        coord.request_stop()
        coord.join(threads)

    return image


def read_records(filename, indices):
    '''
    解析record文件:源文件的图像数据是RGB,uint8,[0,255],一般作为训练数据时,需要归一化到[0,1]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param type:选择图像数据的返回类型
         None:默认将uint8-[0,255]转为float32-[0,255]
         normalization:归一化float32-[0,1]
         centralization:归一化float32-[0,1],再减均值中心化
    :return:
    '''
    # 创建文件队列，不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # 为文件队列创建一个阅读区
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)

    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'data': tf.FixedLenFeature([], tf.string),
        }
    )
    # 获得图像原始的数据
    # pdb.set_trace()
    tf_image = tf.decode_raw(features['data'], tf.uint8)

    images = []
    total_frame = get_example_nums(filename)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for index in range(total_frame):

            image = sess.run(tf_image)
            if index in indices:
                image = image.reshape([180, 320, 3])
                images.append(image)

        coord.request_stop()
        coord.join(threads)

    return images