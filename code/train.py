from v2a_model_clip import V2AModel
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
        # gpus=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        max_epochs=8000,
        check_val_every_n_epoch=100,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    
    model = V2AModel(opt)
    # pdb.set_trace()
    trainset = create_dataset(opt.dataset.metainfo, opt.dataset.train)
    validset = create_dataset(opt.dataset.metainfo, opt.dataset.valid)


    if opt.model.is_continue == True:
        # pdb.set_trace()
        checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
        # checkpoint = 'checkpoints/epoch=1999-loss=0.009001615457236767.ckpt'
        trainer.fit(model, trainset, validset, ckpt_path=checkpoint)
    else: 
        trainer.fit(model, trainset, validset)


if __name__ == '__main__':
    main()