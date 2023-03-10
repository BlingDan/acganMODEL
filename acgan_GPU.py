from glob import glob
import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

batch_size = 32
noise_dim = 100
class_num = 3
epochs = 1000



def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [192, 192]) #调整图像大小
    img = tf.image.random_crop(img, [64, 64, 3]) #随机裁剪成64*64
    img = tf.image.random_flip_left_right(img) #随机左右反转
    img = img / 255.0
    return img, label


#制作数据集
def make_dataset():
    # images_path = glob.glob('F:/wheat_leaf/*/*.jpg') #路径要修改
    # labels = [path.split("\\")[1] for path in images_path] #此处被修改过
    images_path = glob.glob('D:/NotOnlyCode/leaf_image/*/*.jpeg')
    labels = [path.split("\\")[1] for path in images_path]  # 不同设备需要修改
    print(labels)
    random.shuffle(images_path)

    label_to_index = dict((name, index) for index, name in enumerate(np.unique(labels)))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in images_path]
   
    # for image, label in zip(images_path[:5], all_image_labels[:5]):
    #     print(image, "->", label)

    dataset = tf.data.Dataset.from_tensor_slices((images_path, all_image_labels))
    dataset = dataset.map(load_image)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.shuffle(50).batch(batch_size) #小范围的shuffle乱序
    print("输出结果------------》",dataset)

    return dataset


SLoss = tf.keras.losses.BinaryCrossentropy(from_logits=True) #真假损失
CLoss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  #新增分类损失

generator_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

nsample = 10
noise_seed = tf.random.normal([nsample, noise_dim])
label_seed = np.random.randint(0, 3, size=(nsample, 1))

#判别器损失函数
def discrimitor_loss(real_S_out, real_C_out, fake_S_out, label):
    real_loss = SLoss(tf.ones_like(real_S_out), real_S_out)
    real_class_loss = CLoss(label, real_C_out)
    fake_loss = SLoss(tf.zeros_like(fake_S_out), fake_S_out)

    return real_loss + real_class_loss + fake_loss

#生成器损失函数
def geneatoer_loss(fake_S_out, fake_C_out, label):
    fake_loss = SLoss(tf.ones_like(fake_S_out), fake_S_out)
    fake_class_loss = CLoss(label, fake_C_out)

    return fake_loss + fake_class_loss

#生成器
def generator_model():
    noise = tf.keras.layers.Input(shape = ((noise_dim,)))
    label = tf.keras.layers.Input(shape = (()))
    
    x = tf.keras.layers.Embedding(3, 50, input_length=1)(label) #将长度为1的标签映射

    #将x和noise合并，变成长度为150的向量，并希望最终得到
    x = tf.keras.layers.concatenate([noise, x])
    x = tf.keras.layers.Dense(4*4*64*8, use_bias=False)(x)
    x = tf.keras.layers.Reshape((4, 4, 64 * 8))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    #反卷积
    x = tf.keras.layers.Conv2DTranspose(64*4, (5,5), strides=(2,2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(64*2, (5,5), strides=(2,2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    model = tf.keras.Model(inputs=[noise, label], outputs=x)
    return model

#判别器
def discriminator_model():
    image = tf.keras.layers.Input(shape=((64, 64, 3)))

    x = tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same', use_bias=False)(image)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(64*2, (3,3), strides=(2,2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(64*4, (3,3), strides=(2,2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(64*8, (3,3), strides=(2,2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Flatten()(x)

    S_out = tf.keras.layers.Dense(1)(x) #判断01真假
    C_out = tf.keras.layers.Dense(3)(x) #三分类

    model = tf.keras.Model(inputs=image, outputs=[S_out, C_out])
    return model

generator = generator_model()
discriminator = discriminator_model()

#对一个批次的训练函数
def train_step(image, label):
    size = label.shape[0]
    noise = tf.random.normal([size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_taps:
        gen_images = generator((noise, label), training=True)
        fake_S_out, fake_C_out =discriminator(gen_images, training=True)

        real_S_out, real_C_out = discriminator(image, training=True)

        disc_loss = discrimitor_loss(real_S_out, real_C_out, fake_S_out, label)
        gen_loss = geneatoer_loss(fake_S_out, fake_C_out, label)
    
    #计算梯度
    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    dis_grad = dis_taps.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_grad, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(dis_grad, discriminator.trainable_variables))

#训练函数
def train(dataset, epochs):
    print("开始训练")
    for epoch in range(epochs):
        for images_batch, label_batch in dataset:
            train_step(images_batch, label_batch)
        if epoch % 50 == 0: #被修改过，最初为100
            print("epoch:", epoch)
            plot_gen_image(generator,noise_seed, label_seed)

#绘图函数
def plot_gen_image(model, noise, label):
    gen_image = model((noise, label), training=False)
    #gen_image = tf.squeeze(gen_image, -1) #被删掉了

    fig = plt.figure(figsize=(10, 1))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow((gen_image[i, :, :] + 1) / 2, cmap='gray')
        plt.axis('off')
    plt.show()


dataset = make_dataset()
train(dataset,epochs)