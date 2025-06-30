import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# 配置GPU显存按需增长
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf_session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
tf.compat.v1.keras.backend.set_session(tf_session)

# 替换为PTeLU激活函数
class PTeLU(Layer):
    def __init__(self, alpha=1.0, beta=1.0, learnable=False, **kwargs):
        super(PTeLU, self).__init__(**kwargs)
        self.alpha_init = alpha
        self.beta_init = beta
        self.learnable = learnable

    def build(self, input_shape):
        if self.learnable:
            self.alpha = self.add_weight(
                name='alpha',
                shape=(1,),
                initializer=tf.constant_initializer(self.alpha_init),
                trainable=True)
            self.beta = self.add_weight(
                name='beta',
                shape=(1,),
                initializer=tf.constant_initializer(self.beta_init),
                trainable=True)
        else:
            self.alpha = tf.Variable(self.alpha_init, trainable=False, name='alpha')
            self.beta = tf.Variable(self.beta_init, trainable=False, name='beta')
        super(PTeLU, self).build(input_shape)

    def call(self, inputs):
        return self.beta * inputs * tf.math.tanh(tf.math.exp(self.alpha * inputs))

    def get_config(self):
        config = super(PTeLU, self).get_config()
        config.update({
            'alpha': self.alpha_init,
            'beta': self.beta_init,
            'learnable': self.learnable
        })
        return config

# 2. 实现EUCB（高效上采样卷积块）
def EUCB(x, out_channels, kernel_size=3):
    """高效上采样卷积块（Efficient Up-convolution Block）"""
    # 上采样 + 深度可分离卷积
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = DepthwiseConv2D(kernel_size=kernel_size, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = PTeLU(alpha=1.5, beta=0.8, learnable=False)(x)  # 使用PTeLU激活

    # 通道混洗
    def channel_shuffle(x):
        batch_size, height, width, num_channels = tf.unstack(tf.shape(x))
        group_size = 8  # 分组大小
        groups = num_channels // group_size

        # 重塑为 [batch_size, height, width, groups, group_size]
        x = tf.reshape(x, [batch_size, height, width, groups, group_size])
        # 转置为 [batch_size, height, width, group_size, groups]
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        # 重塑回原始形状
        return tf.reshape(x, [batch_size, height, width, num_channels])

    x = Lambda(channel_shuffle)(x)

    # 点卷积通道变换
    x = Conv2D(out_channels, kernel_size=1, strides=1, padding='same', use_bias=True)(x)
    return x

# 3. 实现Universal Inverted Bottleneck (UIB)块
def UIB_block(input_x, out_channels, start_dw_kernel, middle_dw_kernel,
              middle_downsample, stride, expand_ratio):
    """通用倒残差块（Universal Inverted Bottleneck）"""
    in_channels = K.int_shape(input_x)[-1]
    x = input_x

    # 起始深度卷积（可选）
    if start_dw_kernel > 0:
        stride_val = stride if not middle_downsample else 1
        x = DepthwiseConv2D(
            kernel_size=start_dw_kernel,
            strides=stride_val,
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = PTeLU(alpha=1.5, beta=0.8, learnable=False)(x)  # 使用PTeLU激活

    # 扩展通道
    expand_filters = max(8, int(in_channels * expand_ratio))  # 确保至少8个通道
    x = Conv2D(expand_filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = PTeLU(alpha=1.5, beta=0.8, learnable=False)(x)  # 使用PTeLU激活

    # 中间深度卷积（可选）
    if middle_dw_kernel > 0:
        stride_val = stride if middle_downsample else 1
        x = DepthwiseConv2D(
            kernel_size=middle_dw_kernel,
            strides=stride_val,
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = PTeLU(alpha=1.5, beta=0.8, learnable=False)(x)  # 使用PTeLU激活

    # 投影层
    x = Conv2D(out_channels, kernel_size=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # 残差连接
    if stride == 1 and in_channels == out_channels:
        x = Add()([x, input_x])

    return x

# 4. 构建MobileNetV4模型
def build_mobilenetv4(input_shape=(224, 224, 3), num_classes=2, model_type='ConvMedium'):
    """构建MobileNetV4模型"""
    inputs = Input(shape=input_shape)
    x = inputs

    # 初始卷积层
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = PTeLU(alpha=1.5, beta=0.8, learnable=False)(x)

    # 根据模型类型选择配置
    if model_type == 'ConvSmall':
        # Layer 1
        x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = PTeLU(alpha=1.5, beta=0.8, learnable=False)(x)

        x = Conv2D(32, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = PTeLU(alpha=1.5, beta=0.8, learnable=False)(x)

        # Layer 2
        x = Conv2D(96, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = PTeLU(alpha=1.5, beta=0.8, learnable=False)(x)

        x = Conv2D(64, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = PTeLU(alpha=1.5, beta=0.8, learnable=False)(x)

        # Layer 3 (UIB块)
        x = UIB_block(x, 96, 5, 5, True, 2, 3.0)
        for _ in range(4):
            x = UIB_block(x, 96, 0, 3, True, 1, 2.0)
        x = UIB_block(x, 96, 3, 0, True, 1, 4.0)

        # Layer 4 (UIB块)
        x = UIB_block(x, 128, 3, 3, True, 2, 6.0)
        x = UIB_block(x, 128, 5, 5, True, 1, 4.0)
        for _ in range(3):
            x = UIB_block(x, 128, 0, 5, True, 1, 4.0)
        for _ in range(2):
            x = UIB_block(x, 128, 0, 3, True, 1, 4.0)

        # 最终卷积层
        x = Conv2D(960, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = PTeLU(alpha=1.5, beta=0.8, learnable=False)(x)

        x = Conv2D(1280, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = PTeLU(alpha=1.5, beta=0.8, learnable=False)(x)

    elif model_type == 'ConvMedium':
        # Layer 1
        x = Conv2D(48, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = PTeLU(alpha=1.5, beta=0.8, learnable=False)(x)

        # Layer 2
        x = UIB_block(x, 80, 3, 5, True, 2, 4.0)
        x = UIB_block(x, 80, 3, 3, True, 1, 2.0)

        # Layer 3
        x = UIB_block(x, 160, 3, 5, True, 2, 6.0)
        for _ in range(7):
            x = UIB_block(x, 160, 3, 3, True, 1, 4.0)

        # Layer 4
        x = UIB_block(x, 256, 5, 5, True, 2, 6.0)
        for _ in range(10):
            x = UIB_block(x, 256, 3, 5, True, 1, 4.0)

        # 使用EUCB进行最终上采样（替代全局平均池化）
        x = EUCB(x, 256)  # 上采样到更大尺寸

    # 全局平均池化
    x = GlobalAveragePooling2D()(x)

    # 分类头
    x = Dense(1280, activation=PTeLU(alpha=1.5, beta=0.8, learnable=False))(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name=f'MobileNetV4_{model_type}')

# 数据生成器配置
train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1 / 255.0)
valid_datagen = ImageDataGenerator(rescale=1 / 255.0)

# 数据集路径配置
train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    shuffle=True,
)

test_generator = test_datagen.flow_from_directory(
    'testduojuli',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    shuffle=False,
)

valid_generator = valid_datagen.flow_from_directory(
    'valid1',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    shuffle=False,
)

# 构建并编译MobileNetV4模型
model = build_mobilenetv4(model_type='ConvMedium')
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=categorical_crossentropy,
    metrics=['accuracy']
)

# 打印模型结构
model.summary()

# 训练模型
history = model.fit(
    train_generator,
    verbose=2,
    epochs=100,
    validation_data=valid_generator,
    callbacks=[
        CSVLogger('train.log'),
        ReduceLROnPlateau(factor=0.95, patience=3, verbose=2),
        ModelCheckpoint('PTmobilenetv4_model.h5',
                        save_best_only=True,
                        verbose=2,
                        monitor='val_accuracy')
    ]
)

# 在训练结束后使用验证集进行最终验证
print("\n--- 开始验证集评估 ---")
valid_results = model.evaluate(valid_generator, verbose=2)
print(f"验证集评估结果: 损失 = {valid_results[0]:.4f}, 准确率 = {valid_results[1]:.4f}")

# 使用测试集进行最终测试
print("\n--- 开始测试集评估 ---")
test_results = model.evaluate(test_generator, verbose=2)
print(f"测试集评估结果: 损失 = {test_results[0]:.4f}, 准确率 = {test_results[1]:.4f}")
