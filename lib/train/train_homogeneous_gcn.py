import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from lib.datasets import HomogeneousPipelineDataset
from lib.models import HomogeneousGCN
from lib.pl import LightningModule


def train_homogeneous_gcn():
    model = LightningModule(
        model=HomogeneousGCN(in_channels=60, out_channels=2),
        loss=F.mse_loss,
    )
    # TODO: remove hardcoded path. Make configurable
    dataset = HomogeneousPipelineDataset(
        r"C:\Users\Konstantin\PycharmProjects\NIR\dataset\pipeline_dataset",
        direction="undirected",
    )
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15])
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)
    test_loader = DataLoader(test_dataset, batch_size=1024)

    # TODO: remove hardcoded path. Make path relative.
    logger = TensorBoardLogger(
        r"C:\Users\Konstantin\PycharmProjects\NIR\experiments",
        name="default_homogeneous_gcn_undirected_data",
    )
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss", save_last=True, every_n_epochs=1)

    trainer = Trainer(
        max_epochs=1000,
        logger=logger,
        callbacks=[checkpoint_callback, ],
        log_every_n_steps=4,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    train_homogeneous_gcn()
