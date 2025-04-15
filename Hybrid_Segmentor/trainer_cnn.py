from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import get_loaders
import pytorch_lightning as pl
import config
from model import CNNSegmentor
from pytorch_lightning.loggers import TensorBoardLogger
from rich.table import Table
from rich.console import Console

if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="cnn_segmentor")

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/v7_BCEDICE0_2_final",  # Directory to save checkpoints
        filename="cnn-{epoch:02d}-{val_loss:.4f}",  # Checkpoint filename format
        save_top_k=1,  # Save only the best checkpoint
        monitor="val_loss",  # Metric to monitor
        mode="min",  # Save the checkpoint with the minimum validation loss
    )

    train_loader, val_loader, test_loader = get_loaders(
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR, config.VAL_MASK_DIR,
        config.TEST_IMG_DIR, config.TEST_MASK_DIR,
        config.BATCH_SIZE, config.NUM_WORKERS, config.PIN_MEMORY
    )
    model = CNNSegmentor(learning_rate=config.LEARNING_RATE).to(config.DEVICE)
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        max_epochs=config.NUM_EPOCHS,
        precision="16-mixed",
        callbacks=[checkpoint_callback],  # Add the checkpoint callback
    )
    trainer.fit(model, train_loader, val_loader)

    # Load the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model checkpoint: {best_model_path}")
    model = CNNSegmentor.load_from_checkpoint(best_model_path)

    # Run validation and test steps
    val_metrics = trainer.validate(model, val_loader, verbose=False)[0]
    test_metrics = trainer.test(model, test_loader, verbose=False)[0]

    # Display validation metrics
    console = Console()
    val_table = Table(title="Validate metric")
    val_table.add_column("Metric", justify="center")
    val_table.add_column("Value", justify="center")
    for key, value in val_metrics.items():
        val_table.add_row(key, f"{value:.6f}")
    console.print(val_table)

    # Display test metrics
    test_table = Table(title="Test metric")
    test_table.add_column("Metric", justify="center")
    test_table.add_column("Value", justify="center")
    for key, value in test_metrics.items():
        test_table.add_row(key, f"{value:.6f}")
    console.print(test_table)
