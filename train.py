import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam

TRAIN_DATA_PATH = "train"
VAL_DATA_PATH = "validation"
IMG_HEIGHT = 32
IMG_WIDTH = 32
RESCALE = 1. / 255


def model01(num_of_classes):
    """
    Basic traffic signs classification model
    without dropout and smaller number of filters.
    """
    model = Sequential([
        Conv2D(30, (5, 5), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(15, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(500, activation="relu"),
        Dense(num_of_classes, activation="softmax")
    ])
    model.compile(
        Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def model02(num_of_classes):
    """
    Traffic signs classification model
    without dropout and larger number of filters.
    """
    model = Sequential([
        Conv2D(60, (5, 5), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(30, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(500, activation="relu"),
        Dense(num_of_classes, activation="softmax")
    ])
    model.compile(
        Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def model03(num_of_classes):
    """
    Traffic signs classification model
    without dropout, larger number of filters and
    more convolutional layers.
    """
    model = Sequential([
        Conv2D(60, (5, 5), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation="relu"),
        Conv2D(60, (5, 5), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(30, (3, 3), activation="relu"),
        Conv2D(30, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(500, activation="relu"),
        Dense(num_of_classes, activation="softmax")
    ])
    model.compile(
        Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def model04(num_of_classes):
    """
    Traffic signs classification model
    with dropout, larger number of filters and
    more convolutional layers.
    """
    model = Sequential([
        Conv2D(60, (5, 5), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation="relu"),
        Conv2D(60, (5, 5), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(30, (3, 3), activation="relu"),
        Conv2D(30, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(500, activation="relu"),
        Dropout(0.5),
        Dense(num_of_classes, activation="softmax")
    ])
    model.compile(
        Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def batch(data, directory, shuffle=True):
    """
    Take the path to a directory and generate
    batches of augmented data.
    """
    return data.flow_from_directory(
        batch_size=100,
        directory=directory,
        class_mode="sparse",
        shuffle=shuffle,
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )


def basic_augmentation(home):
    """
    Basic augmentation that augments the data
    only by rescaling images.

    Returns:
        A tuple of train and validation augmented data
    """
    train_data_gen_aug = ImageDataGenerator(rescale=RESCALE)
    val_data_gen_aug = ImageDataGenerator(rescale=RESCALE)
    return (
        batch(train_data_gen_aug, home.joinpath(TRAIN_DATA_PATH)),
        batch(val_data_gen_aug, home.joinpath(VAL_DATA_PATH), False)
    )


def extended_augmentation(home):
    """
    Extended augmentation that augments the data by
    rescaling, shearing, zooming, brightening images.

    Returns:
        A tuple of train and validation augmented data
    """
    train_data_gen_aug = ImageDataGenerator(
        rescale=RESCALE,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.5, 1.5],
        height_shift_range=0.1,
        width_shift_range=0.1,
        channel_shift_range=0.2
    )
    val_data_gen_aug = ImageDataGenerator(rescale=RESCALE)
    return (
        batch(train_data_gen_aug, home.joinpath(TRAIN_DATA_PATH)),
        batch(val_data_gen_aug, home.joinpath(VAL_DATA_PATH), False)
    )


def get_histogram_data(path):
    return {
        str(directory): len(list(directory.iterdir()))
        for directory in path.iterdir()
    }


def histogram(data, name, save=False, filename=None):
    plt.figure(figsize=(12, 4))
    plt.bar(list(data.keys()), list(data.values()))
    plt.title(f"Distribution of the {name} data")
    plt.xticks([])
    plt.ylabel("Number of images")
    if save and filename is not None:
        plt.savefig(filename)
    plt.show()


def acc_plot(history, save=False, filename=None):
    epochs_range = range(len(history.epoch))
    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, history.history["accuracy"], label="Training accuracy")
    plt.plot(epochs_range, history.history["val_accuracy"], label="Validation accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and validation accuracy")
    if save and filename is not None:
        plt.savefig(filename)
    plt.show()


def loss_plot(history, save=False, filename=None):
    epochs_range = range(len(history.epoch))
    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, history.history["loss"], label="Training loss")
    plt.plot(epochs_range, history.history["val_loss"], label="Validation loss")
    plt.legend(loc="upper right")
    plt.title("Training and validation loss")
    if save and filename is not None:
        plt.savefig(filename)
    plt.show()


MODELS = {
    1: model01,
    2: model02,
    3: model03,
    4: model04
}

AUGMENTATIONS = {
    "basic": basic_augmentation,
    "extended": extended_augmentation
}


def main(argv):
    parser = argparse.ArgumentParser(
        description="Arguments to configurate script"
    )

    parser.add_argument(
        "--model",
        type=int,
        required=True,
        help="specify which model to train (1, 2, 3 or 4)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=15,
        help="specify number of epochs (default: 15)"
    )

    parser.add_argument(
        "--augmentation",
        type=str,
        required=True,
        help="specify data augmentation ('basic' or 'extended')"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="specify if trained model and plots should be saved"
    )

    args = parser.parse_args(argv[1:])
    if args.model not in (1, 2, 3, 4):
        raise ValueError("Selected proper model number (1, 2, 3 or 4)")
    if args.augmentation not in ("basic", "extended"):
        raise ValueError("Selected proper augmentation ('basic' or 'extended')")

    home = Path(argv[0]).parent
    results = home.joinpath("results") if args.save else None
    if results is not None:
        results.mkdir(parents=True, exist_ok=True)

    train_data_gen, val_data_gen = AUGMENTATIONS[args.augmentation](home)

    model = MODELS[args.model](train_data_gen.num_classes)
    model.summary()
    history = model.fit(
        train_data_gen,
        steps_per_epoch=int(np.ceil(train_data_gen.samples / 100)),
        epochs=args.epochs,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(val_data_gen.samples / 100)),
        verbose=1
    )
    model.evaluate(val_data_gen)
    if args.save:
        model.save(results.joinpath(f"model{args.model:02}-{args.augmentation}.keras"))

    train_hist_data = get_histogram_data(home.joinpath(TRAIN_DATA_PATH))
    histogram(
        train_hist_data,
        "training",
        args.save,
        results.joinpath(f"training-hist.png")
    )
    val_hist_data = get_histogram_data(home.joinpath(VAL_DATA_PATH))
    histogram(
        val_hist_data,
        "validation",
        args.save,
        results.joinpath(f"validation-hist.png")
    )
    acc_plot(history, args.save, results.joinpath(f"model{args.model:02}-{args.augmentation}-acc.png"))
    loss_plot(history, args.save, results.joinpath(f"model{args.model:02}-{args.augmentation}-loss.png"))


if __name__ == "__main__":
    main(sys.argv)
