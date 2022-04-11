import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.applications.resnet import ResNet101
from tensorflow.python.keras.applications.resnet import ResNet152
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2,ResNet101V2,ResNet152V2
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten, Conv2D, Activation
from tensorflow.python.keras.layers import BatchNormalization, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.python.keras.layers import Add, Multiply
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

print('tf version :',tf.__version__)

def Residual_Block(
                    input_tensor,
                    sub_block_num=3,
                    filter_num=512,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    layer_name_prefix=''
                    ):
    x = Conv2D(filter_num, (kernel_size,kernel_size),strides=(strides,strides),padding=padding,name=layer_name_prefix+'_conv_0')(input_tensor)
    x = BatchNormalization(name=layer_name_prefix+'_bn_0')(x)

    for sub_block_i in range(1,sub_block_num):
        x = Activation('relu',name=layer_name_prefix+'_relu_%d' % (sub_block_i-1))(x)
        x = Conv2D(filter_num, (kernel_size,kernel_size),strides=(strides,strides),padding=padding,name=layer_name_prefix+'_conv_%d' % sub_block_i)(x)
        x = BatchNormalization(name=layer_name_prefix+'_bn_%d' % sub_block_i)(x)
    
    add_block = Add(name=layer_name_prefix+'_add_block')([x,input_tensor])
    output_tensor = Activation('relu',name=layer_name_prefix+'_relu_%d' % (sub_block_num-1))(add_block)
    return output_tensor

def Conv_Block(
                x,
                sub_block_num=2,
                filter_num=512,
                kernel_size=3,
                strides=1,
                padding='same',
                layer_name_prefix=''
                ):
    for sub_block_i in range(0,sub_block_num):
        x = Conv2D(filter_num, (kernel_size,kernel_size),strides=(strides,strides),padding=padding,name=layer_name_prefix+'_conv_%d' % sub_block_i)(x)
        x = BatchNormalization(name=layer_name_prefix+'_bn_%d' % sub_block_i)(x)
        x = Activation('relu',name=layer_name_prefix+'_relu_%d' % sub_block_i)(x)
    return x

def Conv_Blocks(x,block_num,sub_block_num,filter_num,backbone_name='VGG16',layer_name_prefix='attention'):
    if backbone_name == 'VGG16' or backbone_name == 'VGG19':
        for block_i in range(block_num):
            x = Conv_Block(
                            x,
                            sub_block_num=sub_block_num,
                            filter_num=filter_num,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            layer_name_prefix='%s_conv_block_%d' % (layer_name_prefix,block_i)
                            )
    elif backbone_name == 'ResNet50' or backbone_name == 'ResNet101' or backbone_name == 'ResNet152' or backbone_name == 'ResNet50V2' or backbone_name == 'ResNet101V2' or backbone_name == 'ResNet152V2':
        for block_i in range(block_num):
            x = Residual_Block(
                                x,
                                sub_block_num=sub_block_num,
                                filter_num=filter_num,
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                layer_name_prefix='%s_residual_block_%d' % (layer_name_prefix,block_i)
                                )
    else:
        print('Invalid BackBone Model Name :',backbone_name)
        return -1
    return x

def get_backbone(
                backbone_name='VGG16',
                input_shape=(224, 224, 3),
                trainable=True,
                weights=None,
                show_structure=False
                ):
    if backbone_name == 'VGG16':
        backbone_base = VGG16(include_top=False,input_shape=input_shape,weights=weights)
    elif backbone_name == 'VGG19':
        backbone_base = VGG19(include_top=False,input_shape=input_shape,weights=weights)
    elif backbone_name == 'ResNet50':
        backbone_base = ResNet50(include_top=False,input_shape=input_shape,weights=weights)
    elif backbone_name == 'ResNet50V2':
        backbone_base = ResNet50V2(include_top=False,input_shape=input_shape,weights=weights)
    elif backbone_name == 'ResNet101':
        backbone_base = ResNet101(include_top=False,input_shape=input_shape,weights=weights)        
    elif backbone_name == 'ResNet101V2':
        backbone_base = ResNet101V2(include_top=False,input_shape=input_shape,weights=weights)
    elif backbone_name == 'ResNet152':
        backbone_base = ResNet152(include_top=False,input_shape=input_shape,weights=weights)        
    elif backbone_name == 'ResNet152V2':
        backbone_base = ResNet152V2(include_top=False,input_shape=input_shape,weights=weights)
    else:
        print('Invalid BackBone Model Name :',backbone_name)
        return -1
    
    extract_layer_name = backbone_base.layers[-1].name
    base_conv_filter_num = backbone_base.layers[-1].output_shape[-1]

    last_layer = backbone_base.get_layer(extract_layer_name)
    backbone_output = last_layer.output
    backbone_input = backbone_base.input
    feature_extractor = Model(backbone_input,backbone_output)

    if show_structure:
        feature_extractor.summary()

    if not trainable:
        print('Fix Weights of BackBone')
        for layer in feature_extractor.layers:
            layer.trainable = False

    return feature_extractor,base_conv_filter_num


def attention_branch(feature_map,backbone_name='VGG16',filter_num=512,category_num=1000):
    if backbone_name == 'VGG16' or backbone_name == 'VGG19':
        block_num = 1
        sub_block_num = 2
    elif backbone_name == 'ResNet50' or backbone_name == 'ResNet101' or backbone_name == 'ResNet152' or backbone_name == 'ResNet50V2' or backbone_name == 'ResNet101V2' or backbone_name == 'ResNet152V2':
        block_num = 3
        sub_block_num = 3
    else:
        print('Invalid BackBone Model Name :',backbone_name)
        return -1

    pre_conv = Conv_Blocks(
                            feature_map,
                            block_num=block_num,
                            sub_block_num=sub_block_num,
                            filter_num=filter_num,
                            backbone_name=backbone_name,
                            layer_name_prefix='attention_pre_conv'
                            )

    bn = BatchNormalization(name='attention_bn')(pre_conv)
    conv_0 = Conv2D(category_num, (1,1), activation='relu',name='attention_conv_0')(bn)
    conv_1 = Conv2D(category_num, (1,1),name='attention_conv_1')(conv_0)
    gap = GlobalAveragePooling2D(name='attention_gap')(conv_1)
    attention_prediction = Activation('softmax',name='prob_score_attention')(gap)
    conv_2_m = Conv2D(1, (1,1),name='attention_conv_2_m')(conv_1)
    bn_m = BatchNormalization(name='attention_bn_m')(conv_2_m)
    attention_map = Activation('sigmoid',name='attention_map')(bn_m)
    return attention_prediction,attention_map

def perception_branch(feature_map_m,backbone_name='VGG16',filter_num=512,category_num=1000,dense_unit=4024):

    if backbone_name == 'VGG16' or backbone_name == 'VGG19':
        flatten = Flatten(name='perception_flatten')(feature_map_m)
        dense_0 = Dense(dense_unit, activation='relu',name='perception_dense_0')(flatten)    
        dense_1 = Dense(dense_unit, activation='relu',name='perception_dense_1')(dense_0)    
        perception_prediction = Dense(category_num, activation='softmax',name='prob_score_perception')(dense_1)
    elif backbone_name == 'ResNet50' or backbone_name == 'ResNet101' or backbone_name == 'ResNet152' or backbone_name == 'ResNet50V2' or backbone_name == 'ResNet101V2' or backbone_name == 'ResNet152V2':
        block_num = 3
        sub_block_num = 3
        pre_conv = Conv_Blocks(
                                feature_map_m,
                                block_num=block_num,
                                sub_block_num=sub_block_num,
                                filter_num=filter_num,
                                backbone_name=backbone_name,
                                layer_name_prefix='perception_pre_conv'
                                )
        ap = AveragePooling2D(pool_size=(2, 2),padding='same',name='perception_ap')(pre_conv)
        perception_prediction = Activation('softmax',name='prob_score_perception')(ap)
    
    else:
        print('Invalid BackBone Model Name :',backbone_name)
        return -1

    return perception_prediction

def ABN_Model(input_shape=(224, 224, 3),category_num=1000,backbone_name='ResNet50',fine_tune=True,gamma=0.5,dense_unit=4024,backbone_trainable=True,show_model_structure=True):
    input_tensor = Input(shape=input_shape)
    feature_extractor,base_conv_filter_num = get_backbone(backbone_name=backbone_name,input_shape=input_shape,trainable=backbone_trainable)
    feature_map = feature_extractor(input_tensor)
    attention_prediction,attention_map = attention_branch(feature_map,backbone_name=backbone_name,filter_num=base_conv_filter_num)
    
    feature_map_m = Multiply()([feature_map,attention_map])
    feature_map_m = Add()([feature_map_m,feature_map])
    perception_prediction = perception_branch(feature_map_m,backbone_name=backbone_name,filter_num=base_conv_filter_num,category_num=category_num,dense_unit=dense_unit)

    model = Model(inputs=input_tensor, outputs=[perception_prediction,attention_prediction,attention_map])

    if show_model_structure:
        model.summary()

    if fine_tune:
        model.compile(
            optimizer=SGD(learning_rate=0.1, momentum=0.9, decay=1e-4),
            loss={
                'categorical_crossentropy':'prob_score_perception',
                'categorical_crossentropy':'prob_score_attention',
                'mean_squared_error':'attention_map'
                },
            loss_weights={
                'prob_score_perception':(1.0 - gamma) * 0.5,
                'prob_score_attention':(1.0 - gamma) * 0.5,
                'attention_map':gamma
                },
            metrics={
                'prob_score_perception':'categorical_accuracy',
                'prob_score_attention':'categorical_accuracy',
                'attention_map':'accuracy'
                }
        )
    else:
        model.compile(
            optimizer=SGD(lr=0.1, momentum=0.9, decay=1e-4),
            loss={
                'categorical_crossentropy':'prob_score_perception',
                'categorical_crossentropy':'prob_score_attention'
                },
            loss_weights={
                'prob_score_perception':0.5,
                'prob_score_attention':0.5
                },
            metrics={
                'prob_score_perception':'categorical_accuracy',
                'prob_score_attention':'categorical_accuracy'
                }
        )

    return model

def make_heatmap(attention_map,origin_img_shape,origin_img=[],mix=True):
    height = origin_img_shape.shape[0]
    width  = origin_img_shape.shape[1]
    attention_img = cv2.resize(attention_map, (width, height), cv2.INTER_LINEAR)
    attention_img = np.maximum(attention_img, 0) 
    attention_img = attention_img / attention_img.max()
    jetcam = cv2.applyColorMap(np.uint8(255 * attention_img), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)

    if mix:
        heatmap = (np.float32(heatmap) + origin_img / 2)
    
    return heatmap.astype(np.uint8)

def make_attention_map_label(kde_result,map_shape):
    height = map_shape.shape[0]
    width  = map_shape.shape[1]

    kde_result = np.clip(kde_result,0.0,1.0)
    attention_img_label = kde_result * 255

    attention_img_label = attention_img_label.astype(np.uint8)
    resized_attention_img_label = cv2.resize(attention_img_label, dsize=(width, height))
    attention_map_label = np.float32(resized_attention_img_label) / 255.
    return attention_map_label

def main():
    print('ABN')
    input_shape = (224, 224, 3)
    backbone_name = 'ResNet50'
    fine_tune = True
    gamma = 0.5
    backbone_trainable = not fine_tune
    show_model_structure=True

    abn_model = ABN_Model(
                        input_shape=input_shape,
                        backbone_name=backbone_name,
                        fine_tune=fine_tune,
                        gamma=gamma,
                        backbone_trainable=backbone_trainable,
                        show_model_structure=show_model_structure
                        )
    
if __name__ == "__main__":
    main()