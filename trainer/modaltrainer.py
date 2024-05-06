
from modal import App, Image, Volume
import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning import Trainer, LightningModule, LightningDataModule 
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


logging.basicConfig(level=logging.INFO)
stdout_logger = logging.getLogger(__name__)
img = Image.debian_slim().pip_install("numpy", "transformers", "datasets", "glob2", "lightning", "tensorboard")
app = App("trainer")
volume = Volume.from_name('bigdata')

class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.as_tensor(mean, dtype=torch.float32))
        self.std = torch.nn.Parameter(torch.as_tensor(std, dtype=torch.float32))
    
    def forward(self, x):
        return (x - self.mean) / self.std

class MoveDim(torch.nn.Module):
    def __init__(self, src, dest):
        super().__init__()
        self.src = src
        self.dest = dest
    
    def forward(self, x):
        return x.moveaxis(self.src, self.dest)

class WeatherConfig:
    def __init__(self, mean=[], std=[], num_inputs=52, num_hidden1=64, num_hidden2=128, num_outputs=2, kernel_size=(3, 3)):
        self.mean = mean
        self.std = std
        self.num_inputs = num_inputs
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size

class WeatherModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.layers = torch.nn.Sequential(
            Normalization(self.hparams.mean, self.hparams.std),
            MoveDim(-1, 1),
            torch.nn.Conv2d(self.hparams.num_inputs, self.hparams.num_hidden1, self.hparams.kernel_size),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(self.hparams.num_hidden1, self.hparams.num_hidden2, self.hparams.kernel_size),
            torch.nn.ReLU(),
            MoveDim(1, -1),
            torch.nn.Linear(self.hparams.num_hidden2, self.hparams.num_outputs),
            torch.nn.ReLU()
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        predictions = self(inputs)
        loss = torch.nn.SmoothL1Loss()(predictions, labels)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        predictions = self(inputs)
        loss = torch.nn.SmoothL1Loss()(predictions, labels)
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
        num_hidden1 = 64  # These can be configured as needed
        num_hidden2 = 128
        num_outputs = 2
        kernel_size = (3, 3)
        
        config = {
            'mean': mean.tolist(), 
            'std': std.tolist(), 
            'num_inputs': num_inputs, 
            'num_hidden1': num_hidden1, 
            'num_hidden2': num_hidden2, 
            'num_outputs': num_outputs, 
            'kernel_size': kernel_size
        }
        return WeatherModel(config)

class WeatherDataModule(LightningDataModule):
    def __init__(self, dataset, batch_size=32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        # Convert dataset to PyTorch tensors and split data here if needed
        # This is a placeholder for actual data handling logic
        data = torch.tensor(self.dataset['inputs'], dtype=torch.float32)
        targets = torch.tensor(self.dataset['labels'], dtype=torch.float32)
        dataset = TensorDataset(data, targets)
        self.train_dataset, self.val_dataset = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


# Constants
EPOCHS = 100
BATCH_SIZE = 512
TRAIN_TEST_RATIO = 0.9
NUM_DATASET_READ_PROC = 16
NUM_DATASET_PROC = os.cpu_count() or 8

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

@app.function(image=img)
def augmented(dataset):
    import numpy as np
    """Augments dataset by rotating and flipping the examples."""
    def augment(values: np.ndarray) -> np.ndarray:
        return np.array([np.rot90(values, k, (1, 2)) for k in range(4)] + [np.flip(np.rot90(values, k, (1, 2)), axis=1) for k in range(4)])

    augmented_data = dataset.map(lambda x: {'inputs': augment(x['inputs'])}, batched=True)
    return augmented_data

@app.function(volumes={"/vol": volume}, image=img, timeout=60*60*24)
def train_model(model_path: str, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, train_test_ratio: float = TRAIN_TEST_RATIO):
    dataset = read_dataset(train_test_ratio)
    # dataset = augmented.remote(dataset['train'])  # Assuming augmentation applies only to training data
    model = WeatherModel.create(dataset['train']['inputs'])

    gpus = torch.cuda.device_count()
    stdout_logger.info(f"GPUs available: {gpus}")

    # Setup Checkpointing and Logger
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(model_path, 'checkpoints'),
        filename='weather-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    logger = TensorBoardLogger(save_dir=os.path.join(model_path, 'logs'))

    # Create Trainer with GPU support
    trainer = Trainer(max_epochs=epochs, logger=logger, callbacks=[checkpoint_callback])

    # Assuming WeatherDataModule is correctly implemented to handle the dataset
    data_module = WeatherDataModule(dataset['train'], batch_size=batch_size)
    trainer.fit(model, datamodule=data_module)
    volume.commit()

@app.local_entrypoint()
def main():
    train_model.remote(model_path='/vol/actual')
