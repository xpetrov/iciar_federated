import os
import wandb

import tensorflow as tf
from tensorflow.keras import Sequential, layers, models,\
    preprocessing, metrics, losses, optimizers, callbacks, applications, Model
from tensorflow_addons.metrics import F1Score

from typing import Dict, Optional, Tuple

WANDB_KEY = '70fc9e3697c791bb8e0bd252af2721808872bce9'

config = {
    'wandb_project': 'qfedadam_0.4_1e-5_epr3',
    'target_model_id': 15,
    # less important ------------------
    'client-side_lr': 1e-7,
    'num_clients': 3,
    'batch_size': 4,
    'seed': 41,
    'f1_average': 'macro',
    'wandb_entity': 'xpetrov'
}

DATA_ROOT = 'D:\\xpetrov\\ICIAR2018_BACH_Challenge\\'
DATASET = 'federated\\balanced_3clients+server\\'


def load_concrete_dataset(site_id: int, subset: str) -> tf.data.Dataset:
    shuffle = True if subset=='train' else False
    return preprocessing.image_dataset_from_directory(
        DATA_ROOT + DATASET + 'site' + str(site_id) + '\\' + subset,
        label_mode='categorical',
        class_names=["Benign", "InSitu", "Invasive", "Normal"],
        batch_size=config['batch_size'],
        image_size=(224, 224),
        shuffle=shuffle,
        seed=config['seed']
    )


def load_dataset(site_id: int) -> Tuple[
    tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    return (
        load_concrete_dataset(site_id, 'train'),
        load_concrete_dataset(site_id, 'valid'),
        load_concrete_dataset(site_id, 'test')
    )


def compute_mean_var(dataset):
    rescale = layers.experimental.preprocessing.Rescaling(scale=1./255)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataset:
        data = rescale(data)
        channels_sum += tf.reduce_mean(data, axis=[0, 1, 2], keepdims=False)
        channels_squared_sum += tf.reduce_mean(data**2, axis=[0, 1, 2])
        num_batches += 1
    mean = channels_sum / num_batches
    var = channels_squared_sum/num_batches - mean**2
    #std = var**0.5
    return mean, var


def build_vgg_model(mean, variance) -> Model:
    preprocessing_layer = Sequential([
        layers.experimental.preprocessing.Rescaling(scale=1./255),
        layers.experimental.preprocessing.Normalization(
            mean=mean, variance=variance),
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(
            0.03, fill_mode='reflect', interpolation='bilinear') # [-3% * 2pi, 3% * 2pi]
    ])

    vgg16_pretrained = applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    vgg16_pretrained.trainable = False

    custom_top = models.Sequential()
    custom_top.add(layers.Flatten())
    custom_top.add(layers.Dense(4096, activation='relu',
              kernel_initializer='he_uniform'))
    custom_top.add(layers.Dense(4096, activation='relu',
              kernel_initializer='he_uniform'))
    custom_top.add(layers.Dense(4, activation='softmax'))

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = preprocessing_layer(inputs)
    #x = applications.vgg16.preprocess_input(x)
    x = vgg16_pretrained(x, training=False)
    outputs = custom_top(x)
    model = Model(inputs, outputs)
    model.summary()
    return model


def test_global_model(client_id):
    if client_id == config['num_clients'] + 1:
        test_set = load_concrete_dataset(client_id, 'test')
        mean = tf.constant([0.6892141, 0.56010664, 0.670267])
        variance = tf.constant([0.03059223, 0.04291156, 0.0250392])  
    else:
        train_set, _, test_set = load_dataset(client_id)
        mean, variance = compute_mean_var(train_set)
    print(f"Mean: {mean}\nVar : {variance}")

    model = build_vgg_model(mean, variance)
    optimizer = optimizers.Adam(config['client-side_lr'])
    model.compile(optimizer=optimizer,
                  loss=losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[
                      metrics.CategoricalAccuracy(),
                      F1Score(num_classes=4, average='macro')])

    wandb.init(project=config['wandb_project'], entity=config['wandb_entity'])
    wandb.config = config

    load_dir = os.path.join("E:\\xpetrov", "models", config['wandb_project'])
    #load_dir = os.path.join("E:\\xpetrov", "models", config['wandb_project'], str(client_id))

    model_id = config['target_model_id']
    model_name = 'model_' + str(model_id) + '.h5'
    filepath = os.path.join(load_dir, model_name)
    model.load_weights(filepath=filepath)

    metrics_dict = model.evaluate(
        test_set, verbose=2, return_dict=True)    
    wandb.log({
        'test_loss': metrics_dict['loss'],
        'test_categorical_accuracy': metrics_dict['categorical_accuracy'],
        'test_f1_score': metrics_dict['f1_score']
    })
    
    wandb.finish(quiet=True)


def main():
    wandb.login(key=WANDB_KEY)
    for i in range(1, config['num_clients']+2):
        test_global_model(i)
    #test_global_model(3)


if __name__ == '__main__':
    main()
