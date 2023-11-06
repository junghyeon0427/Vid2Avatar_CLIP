from v2a_model_test_clip_implicit import V2AModel
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import glob
import pdb

@hydra.main(config_path="confs", config_name="base")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        save_last=True)
    logger = WandbLogger(project=opt.project_name, name=f"{opt.exp}/{opt.run}")

    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        max_epochs=8010,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )

    model = V2AModel(opt)
    checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
    # checkpoint = 'checkpoints/epoch=8000-loss=0.ckpt'
    testset = create_dataset(opt.dataset.metainfo, opt.dataset.test)

    trainer.fit(model, testset, ckpt_path=checkpoint)

if __name__ == '__main__':
    main()