import torch
from torch import nn
import pytorch_lightning as pl
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import dataset
torch.set_float32_matmul_precision('high')
import warnings
warnings.filterwarnings("ignore")

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        
    def forward(self, src):
        # [batch_size, seq_len]  e.g. [32, 47]
        embedded = self.dropout(self.embedding(src)) # [batch_size, seq_len, emb_dim]  e.g. [32, 47, 1000]
        outputs, (hidden, cell) = self.rnn(embedded)  # hidden: [n_layers, batch, hid_dim] 
                                                      # cell:   [n_layers, batch, hid_dim]
       
        return hidden, cell
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
    def forward(self, input, hidden, cell):
        # input: [batch]  e.g. [32]  — one token per batch item

        input = input.unsqueeze(1)
        # [32] → [32, 1]
        # LSTM needs 3D input, seq_len=1 because we process one token at a time
        # encoder didn't need this because it had the full sequence [32, 47]

        embedded = self.dropout(self.embedding(input))
        # [32, 1] → [32, 1, 1000]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # LSTM takes:
        #   - current token embedding  [32, 1, 1000]
        #   - hidden, cell from previous step (or encoder at step 1)
        # produces:
        #   - output:  [32, 1, 512]
        #   - new hidden, cell — updated memory passed to next step

        prediction = self.fc_out(output.squeeze(1))
        # squeeze removes seq_len=1:  [32, 1, 512] → [32, 512]
        # linear maps to vocab size:  [32, 512]    → [32, sql_vocab_size]
        # each of 32 items now has a score for every possible SQL token

        return prediction, hidden, cell
    
class Seq2Seq(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder   = encoder
        self.decoder   = decoder
        self.criterion = nn.CrossEntropyLoss(ignore_index=dataset.sql_vocab.stoi["<pad>"])
        
        # self.init_weights() 

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size     = src.shape[0]
        trg_len        = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs  = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device) 
        hidden, cell = self.encoder(src)
        x = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            x = trg[:, t] if teacher_force else output.argmax(dim=1)

        return outputs
    
    # def init_weights(self):
    #     for name, param in self.named_parameters():
    #         nn.init.uniform_(param.data, -0.08, 0.08)

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = optim.Adam(self.parameters(), lr=1e-3,weight_decay = 1e-4 )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor":   "val_loss",
                "interval":  "epoch",
                "frequency": 1
            }
        }

    def training_step(self, batch, batch_idx):
        src, trg = batch
        output   = self(src, trg)

        output = output[:, 1:, :].reshape(-1, output.shape[-1])
        trg    = trg[:, 1:].reshape(-1)

        loss  = self.criterion(output, trg)
        preds = output.argmax(dim=1)
        acc   = (preds == trg).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc',  acc,  prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch
        output   = self(src, trg, teacher_forcing_ratio=0.0)

        output = output[:, 1:, :].reshape(-1, output.shape[-1])
        trg    = trg[:, 1:].reshape(-1)

        loss  = self.criterion(output, trg)
        preds = output.argmax(dim=1)
        acc   = (preds == trg).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc',  acc,  prog_bar=True)

    def predict_step(self, batch, batch_idx, max_len=50):
        """Bulk inference — called by trainer.predict()"""
        src, _     = batch
        batch_size = src.shape[0]
        end_idx    = dataset.sql_vocab.stoi["<end>"]
        finished   = torch.zeros(batch_size, dtype=torch.bool).to(src.device)
        outputs    = []

        with torch.no_grad():
            hidden, cell = self.encoder(src)
            x = torch.full(
                (batch_size,),
                fill_value = dataset.sql_vocab.stoi["<start>"],
                dtype      = torch.long,
                device     = src.device
            )
            for _ in range(max_len):
                output, hidden, cell = self.decoder(x, hidden, cell)
                x = output.argmax(dim=1)
                outputs.append(x.clone())
                finished |= (x == end_idx)
                if finished.all():
                    break

        preds = torch.stack(outputs, dim=1)             # [batch, seq_len]
        return [dataset.sql_vocab.decode(p) for p in preds]
    

    def translate(self, sentence: str, max_len=50):
        """Single string inference — called directly by you"""
        was_training = self.training      # save current mode
        self.eval()
        tokens  = dataset.text_vocab.encode(sentence).unsqueeze(0).to(self.device)
        end_idx = dataset.sql_vocab.stoi["<end>"]
        result  = []

        with torch.no_grad():
            hidden, cell = self.encoder(tokens)
            x = torch.tensor(
                [dataset.sql_vocab.stoi["<start>"]],
                device = self.device
            )
            for _ in range(max_len):
                output, hidden, cell = self.decoder(x, hidden, cell)
                x     = output.argmax(dim=1)
                token = dataset.sql_vocab.itos[x.item()]
                if token == "<end>":
                    break
                result.append(token)
        
        if was_training:
            self.train()

        return " ".join(result)