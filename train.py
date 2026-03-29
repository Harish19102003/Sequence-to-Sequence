import torch
import os 
from pathlib import Path
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from model import Encoder, Decoder, Seq2Seq
from dataset import Loader
from config import dataset, text_vocab, INPUT_DIM, ENC_EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT, OUTPUT_DIM, DEC_EMB_DIM, batch_size, epochs, grad_clip

g=torch.Generator()
g.manual_seed(42)



train_dataset,test_dataset = random_split(dataset, [0.8, 0.2], generator=g)
train_loader = Loader(train_dataset,pad_idx=text_vocab.stoi["<pad>"], batch_size=batch_size, shuffle=True)
val_loader = Loader(test_dataset, pad_idx=text_vocab.stoi["<pad>"], batch_size=batch_size, shuffle=False)


enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(enc, dec)

def main():
    output_dir = Path("checkpoints")
    if not os.path.exists(output_dir):
        output_dir.mkdir()

    early_stop = EarlyStopping(
        monitor  = 'val_loss',
        patience = 5,        # stop if val_loss doesn't improve for 5 epochs
        mode     = 'min'
    )

    checkpoint = ModelCheckpoint(
        dirpath= output_dir,
        monitor   = 'val_loss',
        save_top_k = 1,          # only keep best model
        mode      = 'min',
        filename  = 'text_to_sql',
        auto_insert_metric_name=False
        )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(max_epochs=epochs, gradient_clip_val=grad_clip, callbacks= [early_stop, checkpoint,lr_monitor])
    trainer.fit(model,train_loader,val_loader)

if __name__ == "__main__":
    main()