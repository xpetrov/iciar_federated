import os
import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
from tensorflow.keras import Sequential, layers, models,\
    preprocessing, metrics, losses, optimizers, callbacks, applications, Model
from tensorflow_addons.metrics import F1Score

from typing import Dict, Optional, Tuple

WANDB_KEY = '70fc9e3697c791bb8e0bd252af2721808872bce9'

config = {
    'num_clients': 3,
    'learning_rate': 1e-7,
    'epochs': 50,
    'batch_size': 4,
    'seed': 41,
    'wandb_project': 'single_institution_model',
    'wandb_entity': 'xpetrov'
}

DATA_ROOT = 'D:\\xpetrov\\ICIAR2018_BACH_Challenge\\'
DATASET = 'federated\\balanced_3clients+server\\'


def main():
    wandb.login(key=WANDB_KEY)
    for site_id in range(1, config['num_clients']+1):
        fit_model(site_id)


def fit_model(site_id):
    train_set = load_concrete_dataset(site_id, 'train')
    valid_set = load_concrete_dataset(site_id, 'valid')
    mean, variance = compute_mean_var(train_set)
    print(f"Mean: {mean}\nVar : {variance}")

    model = build_vgg_model(mean, variance)
    optimizer = optimizers.Adam(config['learning_rate'])
    model.compile(optimizer=optimizer,
                  loss=losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[
                      metrics.CategoricalAccuracy(),
                      F1Score(num_classes=4, average='macro')])
    
    wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'])
    wandb.config = config

    save_dir = os.path.join("E:\\xpetrov", "models", config['wandb_project'], str(site_id))
    os.makedirs(save_dir, exist_ok=True)
    filepath = save_dir + "\\model_{epoch}.h5"

    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        save_freq='epoch'
    )
    
    model.fit(train_set, epochs=config["epochs"],
                        validation_data=valid_set,
                        verbose=2,  # single line per epoch
                        callbacks=[
                            WandbCallback(
                                monitor="val_loss", save_model=False, save_graph=False),
                            model_checkpoint_callback]
                        )
    
    wandb.finish(quiet=True)


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


if __name__ == '__main__':
    main()
