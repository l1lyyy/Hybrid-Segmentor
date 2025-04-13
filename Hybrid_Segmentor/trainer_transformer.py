from dataloader import get_loaders
import pytorch_lightning as pl
import config
from model import TransformerSegmentor
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="transformer_segmentor")
    train_loader, val_loader, _ = get_loaders(
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR, config.VAL_MASK_DIR,
        config.TEST_IMG_DIR, config.TEST_MASK_DIR,
        config.BATCH_SIZE, config.NUM_WORKERS, config.PIN_MEMORY
    )
    model = TransformerSegmentor(learning_rate=config.LEARNING_RATE).to(config.DEVICE)
    trainer = pl.Trainer(logger=logger, accelerator="gpu", max_epochs=config.NUM_EPOCHS, precision="16-mixed")
    trainer.fit(model, train_loader, val_loader)
