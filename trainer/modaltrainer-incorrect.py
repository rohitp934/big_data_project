"""
INCORRECT - DOES NOT WORK
"""

import os
import numpy as np
import torch
import logging
from modal import App, Image, Volume, gpu
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

logging.basicConfig(level=logging.INFO)
stdout_logger = logging.getLogger(__name__)
img = Image.debian_slim().pip_install("numpy", "transformers", "datasets", "glob2", "lightning", "tensorboard")
app = App("trainer3")
volume = Volume.from_name('bigdata')

class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.std = torch.nn.Parameter(torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x) -> torch.Tensor:
        return (x - self.mean) / self.std

class TFNetConfig:
    def __init__(self, mean=[], std=[], num_inputs=13, num_classes=9, num_hidden=64, kernel_size=5):
        self.mean = mean
        self.std = std
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size

class TFNet(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        # Ensure the output has num_classes channels
        self.layers = torch.nn.Sequential(
            Normalization(self.hparams.mean, self.hparams.std),
            torch.nn.Conv2d(self.hparams.num_inputs, self.hparams.num_hidden, self.hparams.kernel_size, padding=self.hparams.kernel_size // 2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(self.hparams.num_hidden, self.hparams.num_classes, self.hparams.kernel_size, padding=self.hparams.kernel_size // 2),
            torch.nn.ReLU()  # Remove Softmax here, CrossEntropyLoss will handle it
        )
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Adjust input from NHWC to NCHW
        return self.layers(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.squeeze(-1).long()  # Ensure labels are [batch_size, height, width]
        predictions = self(inputs)
        loss = torch.nn.functional.cross_entropy(predictions, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.squeeze(-1).long()  # Ensure labels are [batch_size, height, width]
        predictions = self(inputs)
        loss = torch.nn.functional.cross_entropy(predictions, labels)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    @staticmethod
    def create(inputs):
        # Assuming inputs is a numpy array or similar
        data = np.array(inputs, dtype=np.float32)
        mean = data.mean(axis=(0, 1, 2), keepdims=True)
        std = data.std(axis=(0, 1, 2), keepdims=True)

        # Assuming num_inputs equals the number of channels in the input data
        num_inputs = data.shape[-1]
        num_hidden = 64  # These can be configured as needed
        num_classes = 9
        kernel_size = 5

        config = {
            'mean': mean.flatten().tolist(),  # Flatten since it needs to match the dimensionality of a Tensor
            'std': std.flatten().tolist(),
            'num_inputs': num_inputs,
            'num_hidden': num_hidden,
            'num_classes': num_classes,
            'kernel_size': kernel_size
        }
        return TFNet(config)

class TFDataModule(LightningDataModule):
    def __init__(self, dataset, batch_size=32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        inputs = torch.tensor(self.dataset['inputs'], dtype=torch.float32).permute(0, 3, 1, 2)  # Adjust inputs to NCHW
        labels = torch.tensor(self.dataset['labels'], dtype=torch.long).squeeze(-1)  # Remove channel dim from labels if it exists

        tensor_dataset = TensorDataset(inputs, labels)
        train_size = int(len(tensor_dataset) * 0.8)
        val_size = len(tensor_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(tensor_dataset, [train_size, val_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

TRAIN_TEST_RATIO = 0.9
NUM_DATASET_READ_PROC = 16
NUM_DATASET_PROC = os.cpu_count() or 8
EPOCHS = 100
BATCH_SIZE = 512

def read_dataset(train_test_ratio: float):
    from datasets.arrow_dataset import Dataset
    from glob import glob
    import os
    import numpy as np
    
    data_path = "/vol/actual/npz"
    def read_data_file(item: dict[str, str]) -> dict[str, np.ndarray]:
        with open(item['filename'], 'rb') as f:
            npz = np.load(f)
            return {'inputs': npz['inputs'], 'labels': npz['labels']}
    
    def flatten(batch: dict) -> dict:
        return {key: np.concatenate(values) for key, values in batch.items()}
    # stdout_logger.info(os.listdir(data_path))
    files = glob(os.path.join(data_path, '*.npz'))
    files = files[:5]
    # stdout_logger.info(f"pwd: {os.getcwd()}, Files found: {files}")
    dataset = (
        Dataset.from_dict({'filename': files})
        .map(read_data_file, num_proc=NUM_DATASET_READ_PROC, remove_columns=['filename'])
        .map(flatten, batched=True, num_proc=NUM_DATASET_PROC)
    )
    return dataset.train_test_split(train_size=train_test_ratio)


@app.function(volumes = {"/vol": volume}, image=img, gpu=gpu.A10G(count=1), timeout=60*60*24)
def train_model(data_path='/vol/actual/npz'):
    model_path = '/vol/actual'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = read_dataset(TRAIN_TEST_RATIO)

    stdout_logger.info(f"Read the dataset. Dataset: {dataset}\nTraining: {dataset['train']}\test: {dataset['test']}")
    stdout_logger.info(f"Input and Labels: {np.array(dataset['train']['inputs']).shape}, {np.array(dataset['train']['labels']).shape}\n") 
    model = TFNet.create(dataset['train']['inputs'])

    stdout_logger.info(f"Model: {model}")

    data_module = TFDataModule(dataset['train'], batch_size=BATCH_SIZE)
    stdout_logger.info(f"Training data module: {data_module}")
    # Logger and Checkpoints
    logger = TensorBoardLogger(save_dir=os.path.join(model_path, 'logs'))
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    trainer = Trainer(
        max_epochs=EPOCHS,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=-1, # Use all available GPUs
    )
    trainer.fit(model, datamodule=data_module)
    stdout_logger.info(f"Training complete")

@app.local_entrypoint()
def main():
    train_model.remote()