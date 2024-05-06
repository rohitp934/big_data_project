import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from modal import App, Image, Volume, gpu
from pytorch_lightning.callbacks import ModelCheckpoint


# Initialize modal variables
app = App("trainer-imgoingtokillmyself")
volume = Volume.from_name('bigdata')
img = Image.debian_slim().pip_install("numpy", "scikit-learn", "glob2", "lightning", "torchvision")
# Constants
NUM_INPUTS = 3
NUM_CLASSES = 9
BATCH_SIZE = 512
EPOCHS = 100
MEAN = []
STD = []


def one_hot_encode(labels, num_classes):
    # Ensure labels are a PyTorch tensor
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
        
    # Ensure labels are of type torch.long, required for one_hot function
    labels = labels.long()
    
    # Apply one-hot encoding
    one_hot = torch.nn.functional.one_hot(labels[:,:,:,0], num_classes=num_classes)
    
    # one_hot returns a tensor of shape (batch, height, width, num_classes) directly
    return one_hot.permute(0, 3, 1, 2).float()


class NPZDataset(Dataset):
    def __init__(self, image_data, labels, transform=None):
        self.image_data = image_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.image_data.shape[0]

    def __getitem__(self, idx):
        inputs = self.image_data[idx].astype(np.float32)
        labels = self.labels[idx]
        # labels = labels.long().squeeze()
        if self.transform:
            inputs = self.transform(inputs)
        return inputs, labels

def load_data(data_dir, val_size=0.2, test_size=0.1):
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    print(f"Found {len(file_paths)} files: {file_paths}")
    # file_paths = file_paths[:5]
    train_val_paths, test_paths = train_test_split(file_paths, test_size=test_size, random_state=42)
    train_paths, val_paths = train_test_split(train_val_paths, test_size=val_size, random_state=42)
    return train_paths, val_paths, test_paths

def compute_mean_std(images):
    mean = 0.0
    std = 0.0
    for img in images:
        mean += img.mean(axis = (0, 1, 2), keepdims=True)
        std += img.std(axis = (0, 1, 2), keepdims=True)
    
    mean /= images.shape[0]
    std /= images.shape[0]
    return mean, std

class LandCoverModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(NUM_INPUTS, 32, kernel_size=(5,5)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=(5,5)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, NUM_CLASSES, kernel_size = (1,1))
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = self.loss_fn(outputs, labels)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        test_loss = self.loss_fn(outputs, labels)
        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True)
        return {'test_loss': test_loss}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=512, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=512, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=512, shuffle=False, num_workers=4)

@app.function(image=img, timeout=60*60*24, volumes={"/vol": volume}, gpu=gpu.A10G(count=1), _allow_background_volume_commits=True)
def train_model():
    data_dir = '/vol/actual/npz'
    train_paths, val_paths, test_paths = load_data(data_dir)

    model = LandCoverModel()
    train_images = np.concatenate([np.load(f)['inputs'] for f in train_paths], axis=0)
    train_labels = one_hot_encode(np.concatenate([np.load(f)['labels'] for f in train_paths], axis=0), NUM_CLASSES)

    val_images = np.concatenate([np.load(f)['inputs'] for f in val_paths], axis=0)
    val_labels = one_hot_encode(np.concatenate([np.load(f)['labels'] for f in val_paths], axis=0), NUM_CLASSES)

    test_images = np.concatenate([np.load(f)['inputs'] for f in test_paths], axis=0)
    test_labels = one_hot_encode(np.concatenate([np.load(f)['labels'] for f in test_paths], axis=0), NUM_CLASSES)


    mean, std = compute_mean_std(train_images)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    model.train_dataset = NPZDataset(train_images, train_labels, transform=transform)
    model.val_dataset = NPZDataset(val_images, val_labels, transform=transform)
    model.test_dataset = NPZDataset(test_images, test_labels, transform=transform)

    checkpoint_callback = ModelCheckpoint(
        dirpath='/vol/actual/checkpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        auto_insert_metric_name=False
    )

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=-1,
    )
    trainer.fit(model)
    trainer.test(model)

@app.local_entrypoint()
def main():
    train_model.remote()