#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#resnet , vgg , densenet

import tensorflow as tf

def cls_model(args):
    if args.net_type == "resnet":
        if args.depth == "18":
            base_model = tf.keras.applications.ResNet18(include_top=False, weights='imagenet' if args.pretrained else None)
        elif args.depth == "34":
            base_model = tf.keras.applications.ResNet34(include_top=False, weights='imagenet' if args.pretrained else None)
        elif args.depth == "50":
            base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet' if args.pretrained else None)
        elif args.depth == "101":
            base_model = tf.keras.applications.ResNet101(include_top=False, weights='imagenet' if args.pretrained else None)
        elif args.depth == "152":
            base_model = tf.keras.applications.ResNet152(include_top=False, weights='imagenet' if args.pretrained else None)
        else:
            return None

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        predictions = tf.keras.layers.Dense(args.num_class, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    elif args.net_type == "vgg":
        if args.depth == "16":
            base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet' if args.pretrained else None)
        elif args.depth == "19":
            base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet' if args.pretrained else None)
        elif args.depth == "16bn":
            base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet' if args.pretrained else None)
        elif args.depth == "19bn":
            base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet' if args.pretrained else None)
        else:
            return None

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        predictions = tf.keras.layers.Dense(args.num_class, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    elif args.net_type == "densenet":
        if args.depth == "121":
            base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet' if args.pretrained else None)
        elif args.depth == "169":
            base_model = tf.keras.applications.DenseNet169(include_top=False, weights='imagenet' if args.pretrained else None)
        elif args.depth == "201":
            base_model = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet' if args.pretrained else None)
        else:
            return None

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        predictions = tf.keras.layers.Dense(args.num_class, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    elif args.net_type == "inception":
        if args.depth == "v3":
            base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet' if args.pretrained else None)
        else:
            return None

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        predictions = tf.keras.layers.Dense(args.num_class, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    else:
        return None

    return model

