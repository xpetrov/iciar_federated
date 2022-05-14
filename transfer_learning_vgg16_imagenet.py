import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from tensorflow.keras import layers, models,\
    preprocessing, metrics, losses, optimizers, callbacks, applications
import numpy as np

n_runs = 7
sweep_config = {
    'name': 'BACH-centralized-vgg-local',
    'method': 'grid',
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
    },
    'parameters': {
        'batch_size': {'value': 4},
        'epochs': {'value': 50},
        'optimizer': {'value': 'adam'},
        'learning_rate': {'values': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]},
        'dropout_fc_1': {'values': [.0]},
        'dropout_fc_2': {'values': [.0]}
    }
}


def main():
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="vgg16")

    print("Loading dataset...")
    batch_size = 4
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
    # centralized dataset
    #Mean: [0.6892141  0.56010664 0.670267  ]
    #Var : [0.03059223 0.04291156 0.0250392 ]
    #Std : [0.17490636 0.20715106 0.15823779]
    
    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config

            model = build_vgg_model(config, mean=mean, variance=variance)
            optimizer = build_optimizer(config.optimizer, config.learning_rate)

            model.compile(optimizer=optimizer,
                          loss=losses.CategoricalCrossentropy(from_logits=False),
                          metrics=[
                              metrics.CategoricalAccuracy(),
                              F1Score(num_classes=4, average='macro')])

            early_stopping_fn = callbacks.EarlyStopping(
                monitor='val_loss', patience=7)

            model.fit(train_dataset, epochs=config.epochs,
                      validation_data=valid_dataset,
                      verbose=2,  # single line per epoch
                      callbacks=[
                          WandbCallback(monitor="val_loss", save_model=False, save_graph=False),
                          early_stopping_fn])
    
    wandb.agent(sweep_id, train, count=n_runs)


def load_dataset(batch_size, seed=int(np.random.random()*100), validation_split=0.2):
    train_dataset = preprocessing.image_dataset_from_directory(
        'D:\\xpetrov\\ICIAR2018_BACH_Challenge\\centralized\\site1\\train',
        label_mode='categorical',
        class_names=["Benign", "InSitu", "Invasive", "Normal"],
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=True,
        #validation_split=validation_split,
        seed=seed,
        #subset="training"
    )
    valid_dataset = preprocessing.image_dataset_from_directory(
        'D:\\xpetrov\\ICIAR2018_BACH_Challenge\\centralized\\site1\\valid',
        label_mode='categorical',
        class_names=["Benign", "InSitu", "Invasive", "Normal"],
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=True,
        #validation_split=validation_split,
        seed=seed,
        #subset="validation"
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


def build_vgg_model(config, mean, variance):
    preprocessing_layer = tf.keras.Sequential([
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
    custom_top.add(layers.Dropout(rate=config.dropout_fc_1))
    custom_top.add(layers.Dense(4096, activation='relu',
              kernel_initializer='he_uniform'))
    custom_top.add(layers.Dropout(rate=config.dropout_fc_2))
    custom_top.add(layers.Dense(4096, activation='relu',
              kernel_initializer='he_uniform'))
    custom_top.add(layers.Dense(4, activation='softmax'))

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = preprocessing_layer(inputs)
    #x = applications.vgg16.preprocess_input(x)
    x = vgg16_pretrained(x, training=False)
    outputs = custom_top(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model


if __name__ == '__main__':
    main()
