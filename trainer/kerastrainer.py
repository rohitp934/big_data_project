from __future__ import annotations


import tensorflow as tf
import numpy as np
from modal import App, Image, Volume, gpu

# Initialize modal variables
app = App("ktrainer")
volume = Volume.from_name('bigdata')
img = Image.from_registry(
    "tensorflow/tensorflow:2.12.0-gpu",
).pip_install("protobuf==3.20.*", "numpy")

# Default values.
EPOCHS = 100
BATCH_SIZE = 512
KERNEL_SIZE = 5

# Constants.
NUM_INPUTS = 13
NUM_CLASSES = 9
TRAIN_TEST_RATIO = 90  # percent for training, the rest for testing/validation
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 8

def load_npz_file(file_path_tensor):
    """Loads a training example from NPZ files."""
    # Read the file content as bytes
    file_content = tf.io.read_file(file_path_tensor)
    
    # Use a numpy-compatible way to load from buffer
    def numpy_load_from_buffer(buffer):
        from io import BytesIO
        with BytesIO(buffer) as f:
            data = np.load(f, allow_pickle=True)
            # Assuming 'inputs' and 'labels' are the keys in your npz file
            inputs = data['inputs']
            labels = data['labels']
            return inputs, labels

    # Convert the tensor bytes to numpy arrays
    inputs, labels = tf.py_function(numpy_load_from_buffer, [file_content], [tf.float32, tf.uint8])
    
    # Ensure the shapes are set correctly as `py_function` loses shape information
    inputs.set_shape([None, None, NUM_INPUTS])
    labels.set_shape([None, None, 1])

    # One-hot encode the labels
    one_hot_labels = tf.one_hot(tf.squeeze(labels, axis=-1), NUM_CLASSES)
    
    return inputs, one_hot_labels

def read_dataset(data_path: str):
    """Reads NPZ files from a directory into a tf.data.Dataset."""
    file_pattern = tf.io.gfile.glob(data_path + '/*.npz')
    dataset = tf.data.Dataset.from_tensor_slices(file_pattern)
    return dataset.map(load_npz_file, num_parallel_calls=tf.data.AUTOTUNE)

def split_dataset(
    dataset: tf.data.Dataset,
    batch_size: int = BATCH_SIZE,
    train_test_ratio: int = TRAIN_TEST_RATIO,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Splits a dataset into training and validation subsets.

    Args:
        dataset: Full dataset with all the training examples.
        batch_size: Number of examples per training batch.
        train_test_ratio: Percent of the data to use for training.

    Returns: A (training, validation) dataset pair.
    """
    train_size = int(len(dataset) * (train_test_ratio / 100))
    train_dataset = dataset.take(train_size).batch(batch_size).shuffle(SHUFFLE_BUFFER_SIZE)
    validation_dataset = dataset.skip(train_size).batch(batch_size)
    return (train_dataset, validation_dataset)

def create_model(
    dataset: tf.data.Dataset, kernel_size: int = KERNEL_SIZE
) -> tf.keras.Model:
    """Creates a Fully Convolutional Network Keras model.

    Make sure you pass the *training* dataset, not the validation or full dataset.

    Args:
        dataset: Training dataset used to normalize inputs.
        kernel_size: Size of the square of neighboring pixels for the model to look at.

    Returns: A compiled fresh new model (not trained).
    """
    # Adapt the preprocessing layers.
    normalization = tf.keras.layers.Normalization()
    normalization.adapt(dataset.map(lambda inputs, _: inputs))

    # Define the Fully Convolutional Network.
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(None, None, NUM_INPUTS)),
            normalization,
            tf.keras.layers.Conv2D(32, kernel_size, activation="relu"),
            tf.keras.layers.Conv2DTranspose(16, kernel_size, activation="relu"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.OneHotIoU(
                num_classes=NUM_CLASSES,
                target_class_ids=list(range(NUM_CLASSES)),
            )
        ],
    )
    return model

@app.function(image=img, timeout=60*60*24, volumes={"/vol": volume}, gpu=gpu.A10G(count=1))
def run(
    data_path: str = "/vol/actual/npz",
    model_path: str = "/vol/actual",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    kernel_size: int = KERNEL_SIZE,
    train_test_ratio: int = TRAIN_TEST_RATIO,
) -> tf.keras.Model:
    """Creates and trains the model.

    Args:
        data_path: Local or Cloud Storage directory path where the TFRecord files are.
        model_path: Local or Cloud Storage directory path to store the trained model.
        epochs: Number of times the model goes through the training dataset during training.
        batch_size: Number of examples per training batch.
        kernel_size: Size of the square of neighboring pixels for the model to look at.
        train_test_ratio: Percent of the data to use for training.

    Returns: The trained model.
    """
    print(f"data_path: {data_path}")
    print(f"model_path: {model_path}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"kernel_size: {kernel_size}")
    print(f"train_test_ratio: {train_test_ratio}")
    print("-" * 40)

    dataset = read_dataset(data_path)
    (train_dataset, test_dataset) = split_dataset(dataset, batch_size, train_test_ratio)
    print(f"Dataset created with {len(train_dataset)} training examples and {len(test_dataset)} test examples")
    model = create_model(train_dataset, kernel_size)
    print(f"Model created with {model.summary()}")

    class Logger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            print(f"Epoch {epoch + 1}: Loss = {logs.get('loss')}, Accuracy = {logs.get('accuracy')}")


    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=[Logger()]
    )
    model.save(model_path)
    print(f"Model saved to path: {model_path}")
    return model

@app.local_entrypoint()
def main():
    run.remote()