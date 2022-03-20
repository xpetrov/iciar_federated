import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from tensorflow.keras import layers, models,\
    preprocessing, metrics, losses, optimizers, callbacks
import numpy as np

sweep_config = {
    'name': 'BACH-centralized',
    'method': 'grid',
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
    },
    'parameters': {
        'batch_size': {'value': 4},
        'epochs': {'value': 50},
        'optimizer': {'value': 'adam'},
        'learning_rate': {'value': 1e-4},
        'dropout_fc_1': {'values': [.0, .25, .50]},
        'dropout_fc_2': {'values': [.0, .25, .50]}
    },
    # 'early_terminate': {
    #    'type': 'hyperband',
    #    'min_iter': 3,
    #    'eta': 3
    # }
}


def main():
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="bach_centralized_keras")
    
    print("Loading dataset...")
    batch_size = sweep_config['parameters']['batch_size']['value']
    train_dataset, valid_dataset = load_dataset(batch_size=batch_size, seed=42)
    
    print("Checking data distribution for training...")
    print(get_data_distribution(train_dataset))
    print("Checking data distribution for validation...")
    print(get_data_distribution(valid_dataset))
    
    print("Computing mean-var-std...")
    mean, variance, std = get_mean_var_std(train_dataset)
    print(f"Mean: {mean}")
    print(f"Var : {variance}")
    print(f"Std : {std}")

    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config

            model = build_model(config, mean=mean, variance=variance)
            optimizer = build_optimizer(config.optimizer, config.learning_rate)

            model.compile(optimizer=optimizer,
                          loss=losses.CategoricalCrossentropy(from_logits=False),
                          metrics=[
                              metrics.CategoricalAccuracy(),
                              F1Score(num_classes=4, average='macro')])

            early_stopping_fn = callbacks.EarlyStopping(
                monitor='val_loss', patience=5)

            model.fit(train_dataset, epochs=config.epochs,
                      validation_data=valid_dataset,
                      verbose=2,  # single line per epoch
                      callbacks=[
                          WandbCallback(monitor="val_loss"),
                          early_stopping_fn])

    wandb.agent(sweep_id, train, count=9)


def load_dataset(batch_size, seed=int(np.random.random()*100), validation_split=0.2):
    train_dataset = preprocessing.image_dataset_from_directory(
        'D:\\BACH_dataset\\ICIAR2018_BACH_Challenge\\Photos_augmented_png',
        label_mode='categorical',
        class_names=["Benign", "InSitu", "Invasive", "Normal"],
        batch_size=batch_size,
        image_size=(512, 512),
        shuffle=True,
        validation_split=validation_split,
        seed=seed,
        subset="training"
    )
    valid_dataset = preprocessing.image_dataset_from_directory(
        'D:\\BACH_dataset\\ICIAR2018_BACH_Challenge\\Photos_augmented_png',
        label_mode='categorical',
        class_names=["Benign", "InSitu", "Invasive", "Normal"],
        batch_size=batch_size,
        image_size=(512, 512),
        shuffle=True,
        validation_split=validation_split,
        seed=seed,
        subset="validation"
    )
    return train_dataset, valid_dataset


def get_mean_var_std(dataset):
    rescale = layers.experimental.preprocessing.Rescaling(scale=1./255)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataset:
        data = rescale(data)
        channels_sum += tf.reduce_mean(data, axis=[0, 1, 2], keepdims=False)
        channels_squared_sum += tf.reduce_mean(data**2, axis=[0, 1, 2])
        num_batches += 1
    mean = channels_sum / num_batches
    var = channels_squared_sum/num_batches - mean**2
    std = var**0.5
    return mean, var, std


def get_data_distribution(dataset):
    hist = tf.constant([0,0,0,0], dtype=tf.int32)
    for _, labels in dataset:
        labels_flattened = tf.cast(tf.argmax(labels, axis=1), dtype=tf.int32)
        hist = tf.add(
            hist,
            tf.histogram_fixed_width(labels_flattened, [0, 3], nbins=4))
    hist_normalized = tf.divide(hist, tf.reduce_sum(hist))
    return hist.numpy(), hist_normalized.numpy()


def build_model(config, mean, variance):
    # Preprocessing layer
    preprocessing_layer = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(scale=1./255),
        layers.experimental.preprocessing.Normalization(
            mean=mean, variance=variance),
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")
    ])

    # Araujo Net
    model = models.Sequential()
    model.add(layers.Conv2D(16, 3, activation='relu',
              kernel_initializer='he_uniform', input_shape=(512, 512, 3)))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(32, 3, activation='relu',
              kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, 3, activation='relu',
              padding='same', kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, 3, activation='relu',
              padding='same', kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(32, 3, activation='relu',
              kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=config.dropout_fc_1))
    model.add(layers.Dense(256, activation='relu',
              kernel_initializer='he_uniform'))
    model.add(layers.Dropout(rate=config.dropout_fc_2))
    model.add(layers.Dense(128, activation='relu',
              kernel_initializer='he_uniform'))
    model.add(layers.Dense(4, activation='softmax'))

    # Creates a model (preprocessing layer + Araujo Net)
    inputs = tf.keras.Input(shape=(512, 512, 3))
    x = preprocessing_layer(inputs)
    outputs = model(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def build_optimizer(name, learning_rate):
    optim = None
    if name == "adadelta":
        optim = optimizers.Adadelta(
            learning_rate=learning_rate,
            rho=0.9,
            epsilon=1e-6)
    elif name == "adam":
        optim = optimizers.Adam(
            learning_rate=learning_rate
        )
    print(optim.get_config())
    return optim


if __name__ == '__main__':
    main()
