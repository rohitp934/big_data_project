# %%
import os
from glob import glob
from typing import Tuple, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf
from flax.metrics import tensorboard

# from flax.core import FrozenDict
from flax.training import train_state, checkpoints
from modal import App, Image, Volume, gpu
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# %%
# Config
NUM_CLASSES = 9
NUM_INPUTS = 3
KERNEL_SIZE = 5

tf.config.experimental.set_visible_devices([], "GPU")

app = App("flax-climate-forecast")
volume = Volume.from_name("climate-forecast")
img = Image.debian_slim().pip_install(
    "flax",
    "numpy",
    "tensorflow[and-cuda]",
    "tensorboard",
    "tqdm",
    "ml-collections",
    "tensorrt",
)

img = img.run_commands(
    [
        "pip install -U 'jax[cuda12_pip]' -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'",
        "python -m site",
        "pip list | grep nvidia",
        "export PATH=/usr/local/cuda-12/bin:$PATH",
        "export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:/usr/local/lib/python3.11/site-packages/tensorrt_libs/:$LD_LIBRARY_PATH:",
    ]
)


# %%
def read_example(serialized: bytes) -> Tuple[jax.Array, jax.Array]:
    """Parses and reads a training example from bytes.

    Args:
        serialized: Serialized example bytes.

    Returns: An (inputs, labels) pair of arrays.
    """
    npz = np.load(serialized)
    inputs = npz["inputs"]
    labels_landcover = npz["labels_landcover"]
    labels_lst = npz["labels_lst"]

    return (inputs, labels_landcover, labels_lst)


# %%
def interpolate_invalid_output_temperatures(temperatures, valid_range=(200, 330)):
    """Interpolate temperatures outside the valid range using Gaussian filtering."""
    invalid_mask = (temperatures < valid_range[0]) | (temperatures > valid_range[1])
    temperatures_filtered = gaussian_filter(temperatures, sigma=1)
    temperatures[invalid_mask] = temperatures_filtered[invalid_mask]
    return temperatures


# %%
def interpolate_invalid_temperatures(data, valid_range=(200, 330), band_index=2):
    """Interpolate temperatures outside the valid range using Gaussian filtering."""
    errs = 0
    for i in range(data.shape[0]):
        invalid_mask = (data[i, :, :, band_index] < valid_range[0]) | (
            data[i, :, :, band_index] > valid_range[1]
        )
        if np.any(invalid_mask):  # Only apply filtering if there are any invalid values
            errs += 1
            valid_temperatures = gaussian_filter(data[i, :, :, band_index], sigma=1)
            interpolated_values = np.where(
                invalid_mask, valid_temperatures, data[i, :, :, band_index]
            )
            data[i, :, :, band_index] = np.clip(
                interpolated_values, valid_range[0], valid_range[1]
            )
    return data


# %%
def read_dataset(
    data_path: str, train_test_ratio: float
) -> Tuple[Tuple[jax.Array, jax.Array], Tuple[jax.Array, jax.Array]]:
    files = glob(os.path.join(data_path, "*.npz"))
    # files = files[:2]
    # Load data from npz files
    inputs_list = []
    lc_label_list = []
    lst_label_list = []
    for file in files:
        with open(file, "rb") as f:
            inputs, labels_landcover, labels_lst = read_example(f)
            inputs = interpolate_invalid_temperatures(inputs)
            labels_lst = interpolate_invalid_temperatures(labels_lst, band_index=0)
            inputs_list.append(inputs)
            lc_label_list.append(labels_landcover)
            lst_label_list.append(labels_lst)

    # Concatenate data
    inputs = np.concatenate(inputs_list, axis=0)
    labels_landcover = np.concatenate(lc_label_list, axis=0)
    labels_lst = np.concatenate(lst_label_list, axis=0)
    print(
        f"Inputs: {inputs.shape}, Labels Landcover: {labels_landcover.shape}, Labels LST: {labels_lst.shape}"
    )

    train_size = int(inputs.shape[0] * train_test_ratio)
    train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
    train_labels_landcover, test_labels_landcover = (
        labels_landcover[:train_size],
        labels_landcover[train_size:],
    )
    train_labels_lst, test_labels_lst = labels_lst[:train_size], labels_lst[train_size:]

    print(
        f"Training data: {train_inputs.shape}, Landcover: {train_labels_landcover.shape}, LST: {train_labels_lst.shape}"
    )
    print(
        f"Testing data: {test_inputs.shape}, Landcover: {test_labels_landcover.shape}, LST: {test_labels_lst.shape}"
    )

    return (train_inputs, train_labels_landcover, train_labels_lst), (
        test_inputs,
        test_labels_landcover,
        test_labels_lst,
    )


# %%
# x, y = read_dataset("../data/v2/climate_change/", 0.9)


# %%
# Define the Fully Convolutional Network.
class CNN_LandCover(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(KERNEL_SIZE, KERNEL_SIZE))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=16, kernel_size=(KERNEL_SIZE, KERNEL_SIZE))(x)
        x = nn.relu(x)
        x = nn.Dense(features=NUM_CLASSES)(x)

        return x


# %%
class CNN_LST(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(KERNEL_SIZE, KERNEL_SIZE))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=16, kernel_size=(KERNEL_SIZE, KERNEL_SIZE))(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        x = nn.relu(x)  # No negative temperatures (since it is in Kelvin)

        return x


# %%
@jax.jit
def apply_lc(state, images, lc):
    """Computes gradients, loss and accuracy for a single batch."""
    # print(f"images shape: {images.shape}, lc shape: {lc.shape}")
    one_hot = jax.nn.one_hot(lc[:, :, :, -1], NUM_CLASSES)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        loss = optax.losses.softmax_cross_entropy(
            logits=logits, labels=one_hot
        ).mean()  # Softmax Cross Entropy for Classification
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy_c = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(one_hot, -1))

    return grads, loss, accuracy_c, logits


# %%
@jax.jit
def apply_lst(state, images, lst):
    """Computes gradients, loss, and accuracy for a single batch."""

    def loss_fn(params):
        """Calculate loss based on parameters."""
        # Generate logits based on current parameters.
        logits = state.apply_fn({"params": params}, images)
        # Compute mean squared error loss.
        loss = optax.losses.squared_error(predictions=logits, targets=lst).mean()
        return loss, logits

    # Compute gradients and loss, grads needs to be based on params directly influencing loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    # We calculate logits again for returning, which is not efficient but necessary for the return
    # logits = state.apply_fn({"params": state.params}, images)

    return grads, loss, None, logits


# %%
@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


# %%
def train_epoch(state, train_ds, batch_size, rng, label: Literal["lc", "lst"]):
    """Train for a single epoch."""
    train_ds_size = len(train_ds[0])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds[0]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = jnp.array(train_ds[0][perm, ...], dtype=jnp.float32)
        batch_images = jax.nn.standardize(batch_images)

        if label == "lc":
            batch_labels = jnp.array(train_ds[1][perm, ...], dtype=jnp.uint8)
            # print(f"Batch images shape: {batch_images.shape}, Batch labels shape: {batch_labels.shape}")
            grads, loss, acc, _ = apply_lc(state, batch_images, batch_labels)
        else:
            batch_labels = jnp.array(train_ds[2][perm, ...], dtype=jnp.float32)
            grads, loss, acc, _ = apply_lst(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        if label == "lc":
            epoch_accuracy.append(acc)
    train_loss = np.mean(epoch_loss)
    train_accuracy = None
    if label == "lc":
        train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


# %%
def create_train_state(rng, config, label: Literal["lc", "lst"]):
    """Creates initial `TrainState`."""
    if label == "lc":
        model = CNN_LandCover()
    elif label == "lst":
        model = CNN_LST()
    else:
        raise ValueError(f"Unknown label: {label}")
    params = model.init(
        rng, jnp.ones([1, config.img_size, config.img_size, NUM_INPUTS])
    )["params"]
    tx = optax.adam(config.learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def save_predictions(epoch, images, labels, preds, save_dir, label_type):
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}/{label_type}")
    os.makedirs(epoch_dir, exist_ok=True)

    np.save(os.path.join(epoch_dir, "images.npy"), images)
    np.save(os.path.join(epoch_dir, "labels.npy"), labels)
    np.save(os.path.join(epoch_dir, "preds.npy"), preds)


@app.function(
    image=img,
    timeout=60 * 60 * 24,
    volumes={"/vol": volume},
    gpu=gpu.A100(count=1),
    _allow_background_volume_commits=True,
)
def train_and_evaluate(
    config: ml_collections.ConfigDict,
    data_dir: str,
    work_dir: str,
    ckpt_dir: str,
    label: Literal["lc", "lst"],
    test_save_dir: str,
) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      work_dir: Directory where the tensorboard summaries are written to.

    Returns:
      The train state (which includes the `.params`).
    """
    import os
    import shutil

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    shutil.rmtree(ckpt_dir, ignore_errors=True)

    # ckpt_options = ocp.CheckpointManagerOptions(
    #     max_to_keep=3,
    # )
    # ckpt_manager = ocp.CheckpointManager(
    #     ocp.test_utils.erase_and_create_empty(ckpt_dir),
    #     options=ckpt_options,
    # )

    print(f"JAX process: {jax.process_index()} / {jax.process_count()}")
    print(f"JAX local devices: {jax.local_devices()}")
    train_ds, test_ds = read_dataset(data_dir, config.train_test_split)
    rng = jax.random.key(0)

    summary_writer = tensorboard.SummaryWriter(work_dir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config, label)

    test_images = jnp.array(test_ds[0], dtype=jnp.float32)
    test_images = jax.nn.standardize(test_images)

    if label == "lc":
        test_labels = jnp.array(test_ds[1], dtype=jnp.uint8)
    elif label == "lst":
        test_labels = jnp.array(test_ds[2], dtype=jnp.float32)
    else:
        raise ValueError(f"Unknown label: {label}")

    for epoch in tqdm(range(config.num_epochs)):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(
            state, train_ds, config.batch_size, input_rng, label
        )

        if label == "lc":
            _, test_loss, test_accuracy, logits = apply_lc(
                state, test_images, test_labels
            )
        elif label == "lst":
            _, test_loss, test_accuracy, logits = apply_lst(
                state, test_images, test_labels
            )
        else:
            raise ValueError(f"Unknown label: {label}")

        print(f"epoch:{epoch}, train_loss: {train_loss}, test_loss: {test_loss}")
        if label == "lc":
            print(
                f"epoch:{epoch}, train_accuracy: {train_accuracy * 100}, test_accuracy: {test_accuracy * 100}"
            )
            summary_writer.scalar("train_accuracy", train_accuracy, epoch)
            summary_writer.scalar("test_accuracy", test_accuracy, epoch)

        summary_writer.scalar("train_loss", train_loss, epoch)
        summary_writer.scalar("test_loss", test_loss, epoch)

        checkpoints.save_checkpoint(ckpt_dir, state, epoch, prefix="", keep=3)
        if epoch % 10 == 0:
            # Save test preds
            save_predictions(
                epoch, test_images, test_labels, logits, test_save_dir, label
            )
        # ckpt = {"model": state}
        # ckpt_manager.save(epoch, args=ocp.args.StandardSave(ckpt))
        # ckpt_manager.wait_until_finished()

    summary_writer.flush()
    volume.commit()
    return state


# %%
config = ml_collections.ConfigDict()

config.learning_rate = 0.001
config.batch_size = 16
config.num_epochs = 250
config.img_size = 128
config.train_test_split = 0.9


# %%
@app.local_entrypoint()
def main():
    train_and_evaluate.remote(
        config,
        "/vol/v2/data/",
        "/vol/v2/flax/lst/logs",
        "/vol/v2/flax/lst/checkpoints",
        "lst",
        "/vol/v2/flax/lst/test",
    )
