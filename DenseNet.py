
from keras.layers import AveragePooling2D, Input, GlobalMaxPooling2D
from keras import layers, models
from keras import backend


def dense_block(x, blocks, growth_rate, name):
    for i in range(blocks):
        x = conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
#         nChannels = nChannels + growthRate
        
    return x

# BN -> ReLU -> Conv(1x1x(32*4)) -> Dropout(default == False) -> BN -> ReLU -> Conv(3x3x32) -> Dropout -> Concat(identity, net)
def conv_block(x, growth_rate, name):
#     bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    bn_axis = 3 
    
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)

    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)

    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)

    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)

    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)

    x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)

    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    
    return x

# (not last)  BN -> ReLU -> Conv(1x1 x nOutChannels)
# (last)      BN -> ReLU -> AvgPool -> Reshape(nChannels) -> AvgPool
def transition_block(x, reduction, name):
#     bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    bn_axis = 3 

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)

    x = layers.Activation('relu', name=name + '_relu')(x)

    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1, use_bias=False, name=name + '_conv')(x)
                      
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    
    return x


def DenseNet(depth=None,
            custom_blocks = None,
             growthRate = None,
             include_top = None,
             data_set = None,
             input_tensor = None,
             input_shape = None,
             pooling = None,
             classes = None,
             **kwargs):

    assert depth is not None, "'depth' is None"
    assert input_shape is not None, "'input_shape' is None"
    assert data_set is not None, "'data_set' is 'cifar10' or 'imagenet' or 'kaggle' or 'caltech'"
    assert classes is not None, "'classes' is None"
    assert pooling is not None, "'pooling' is None"
    assert include_top is not None, "'include_top' is None"


    
    if data_set is 'cifar10':
        if growthRate is None:
            growthRate = 12
        if classes is None:
            classes = 10

    elif data_set is 'imagenet' or data_set is 'kaggle' or data_set is 'caltech':
        if growthRate is None:
            growthRate = 32
        if classes is None:
            classes = 1000
        if depth is type(list):
            custom_blocks = True

 


    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
        
    bn_axis = 3
    nChannels = 2 * growthRate
    N = (depth - 4)//3
    N = N//2

    if data_set is 'cifar10':
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(img_input)
        x = layers.Conv2D(nChannels, 3, strides=1, use_bias=False, name='conv1/conv')(x)

        # [16, 16, 16]

        x = dense_block(x, N, growthRate, name='conv3')
        x = transition_block(x, 0.5, name='pool3')

        x = dense_block(x, N, growthRate, name='conv4')
        x = transition_block(x, 0.5, name='pool4')

        x = dense_block(x, N, growthRate, name='conv5')

        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
        x = layers.Activation('relu', name='relu')(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = layers.Dense(classes, activation='softmax', name='fc10')(x)
        else:
            if pooling == 'avg':
                x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            elif pooling == 'max':
                x = layers.GlobalMaxPooling2D(name='max_pool')(x)


        # Create model.
        model = models.Model(img_input, x, name=f'densenet{depth}')
        
    elif data_set is 'imagenet' or data_set is 'kaggle' or data_set is 'caltech':
        
        if custom_blocks is None:

            if depth == 121:
                blocks = [6, 12, 24, 16]
            elif depth == 169:
                blocks = [6, 12, 32, 32]
            elif depth == 201:
                blocks = [6, 12, 48, 32]
            elif depth == 161:
                blocks = [6, 12, 36, 24]
            else:
                print(f'{depth} is not allowed depth, you will use custom_blocks parameter')

        elif custom_blocks is not None:

            if len(custom_blocks) == 4:
                blocks = custom_blocks
                depth = sum(blocks) * 2 + 5

            else:
                print(f'custom_blocks is {len(custom_blocks)} length')

        
    
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
        x = layers.Conv2D(nChannels, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
        x = layers.Activation('relu', name='conv1/relu')(x)

        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

        # [6, 12, 48, 32]
        x = dense_block(x, blocks[0], growthRate, name='conv2')
        x = transition_block(x, 0.5, name='pool2')

        x = dense_block(x, blocks[1], growthRate, name='conv3')
        x = transition_block(x, 0.5, name='pool3')

        x = dense_block(x, blocks[2], growthRate, name='conv4')
        x = transition_block(x, 0.5, name='pool4')

        x = dense_block(x, blocks[3], growthRate, name='conv5')

        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
        x = layers.Activation('relu', name='relu')(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
        else:
            if pooling == 'avg':
                x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            elif pooling == 'max':
                x = layers.GlobalMaxPooling2D(name='max_pool')(x)


        # Create model.
        # if blocks == [6, 12, 24, 16]:
        #     model = models.Model(img_input, x, name='densenet121')
        # elif blocks == [6, 12, 32, 32]:
        #     model = models.Model(img_input, x, name='densenet169')
        # elif blocks == [6, 12, 48, 32]:
        #     model = models.Model(img_input, x, name='densenet201')
        # else:
        #     model = models.Model(img_input, x, name='densenet')
        model = models.Model(img_input, x, name=f'densenet{depth}')

    else:
        print(f'{data_setf} is not allowed data_set')

    return model



