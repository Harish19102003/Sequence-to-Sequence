import torch
from torch import nn
import pytorch_lightning as pl
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import dataset, pad_idx, text_vocab, sql_vocab, parse_schema
from config import enc_emb_dim, hid_dim, num_layers, dropout, dec_emb_dim, attention, bidirectional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, attention=None, bidirectional=None):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_bidirectional = bidirectional is not None
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, bidirectional=self.use_bidirectional)
        self.use_attention = attention is not None

        if self.use_bidirectional:
            # hid_dim*2 because forward + backward concatenated
            self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
            self.fc_cell   = nn.Linear(hid_dim * 2, hid_dim)
        
    def forward(self, src):
        # [batch_size, seq_len]  e.g. [32, 47]
        embedded = self.dropout(self.embedding(src)) # [batch_size, seq_len, emb_dim]  e.g. [32, 47, 1000]
        outputs, (hidden, cell) = self.rnn(embedded)  # hidden: [n_layers, batch, hid_dim] 
                                                      # cell:   [n_layers, batch, hid_dim]

        if self.use_bidirectional:
            # ── merge forward and backward states ──
            # hidden: [n_layers*2, batch, hid_dim]
            #       → [n_layers, 2, batch, hid_dim]
            hidden = hidden.view(self.rnn.num_layers, 2, hidden.shape[1], -1)
            cell   = cell.view(self.rnn.num_layers, 2, cell.shape[1], -1)

            # concat fwd and bwd → [n_layers, batch, hid_dim*2]
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
            cell   = torch.cat([cell[:, 0],   cell[:, 1]],   dim=2)

            # project back to hid_dim for decoder compatibility
            hidden = torch.tanh(self.fc_hidden(hidden))  # [n_layers, batch, hid_dim]
            cell   = torch.tanh(self.fc_cell(cell))      # [n_layers, batch, hid_dim]

       
        if self.use_attention:
            return outputs, hidden, cell
        else:
            return hidden, cell
    
class Attention(nn.Module):
    def __init__(self, hid_dim, bidirectional=None):
        super().__init__()
   
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        if bidirectional:
            self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v    = nn.Linear(hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hid_dim]
        # encoder_outputs: [batch, seq_len, hid_dim]

        batch_size = encoder_outputs.shape[0]
        seq_len    = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # [batch, hid_dim] → [batch, 1, hid_dim] → [batch, seq_len, hid_dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # concat: [batch, seq_len, hid_dim * 2]
        # attn:   [batch, seq_len, hid_dim]

        attention = self.v(energy).squeeze(2)
        # v:         [batch, seq_len, 1]
        # squeeze:   [batch, seq_len]

        return torch.softmax(attention, dim=1)
        # attention weights: [batch, seq_len]
        # attention weights sum to 1 across the input sequence, so we can use them to take a weighted average of encoder outputs
        # softmax over seq_len gives us a distribution of attention weights across the input sequence

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention=None, bidirectional=None):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_attention = attention is not None
        self.use_bidirectional = bidirectional is not None

        if self.use_attention:
            self.attention = Attention(hid_dim, bidirectional=bidirectional)
            if self.use_bidirectional:
                # input = embedding + context
                # context is hid_dim*2 (bidirectional encoder output)
                self.rnn    = nn.LSTM(
                    emb_dim + hid_dim * 2, hid_dim, n_layers,   # ← was emb_dim + hid_dim
                    batch_first = True
                )

                # prediction from: rnn output + context + embedding
                self.fc_out = nn.Linear(
                    hid_dim + hid_dim * 2 + emb_dim, output_dim  # ← was hid_dim*2 + emb_dim
                )
            else:
                self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, batch_first=True)
                self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)

        else:
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
            self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs=None):
        # input: [batch]  e.g. [32]  — one token per batch item
        # hidden: [n_layers, batch, hid_dim]
        # cell:   [n_layers, batch, hid_dim]
        # encoder_outputs: [batch, seq_len, hid_dim]
        
        input = input.unsqueeze(1)
        # [32] → [32, 1]
        # LSTM needs 3D input, seq_len=1 because we process one token at a time
        # encoder didn't need this because it had the full sequence [32, 47]

        embedded = self.dropout(self.embedding(input))
        # [32, 1] → [32, 1, emb_dim]

        if self.use_attention and encoder_outputs is not None:
        

            attn_weights = self.attention(hidden[-1], encoder_outputs)
            # hidden[-1]: [batch, hid_dim] — last layer's hidden state
            # attention weights: [batch, seq_len]

            attn_weights = attn_weights.unsqueeze(1)
            # [batch, seq_len] → [batch, 1, seq_len]

            context = torch.bmm(attn_weights, encoder_outputs)
            # batch matmul: [batch, 1, seq_len] bmm [batch, seq_len, hid_dim] → [batch, 1, hid_dim]

            rnn_input = torch.cat((embedded, context), dim=2)
            # concat on feature dimension: [batch, 1, emb_dim + hid_dim]

            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
            # LSTM takes:
            #   - current token embedding + context vector:  [batch, 1, emb_dim + hid_dim]
            #   - hidden, cell from previous step (or encoder at step 1)
            # produces:
            #   - output:  [batch, 1, hid_dim]
            #   - new hidden, cell — updated memory passed to next step

            embedded = embedded.squeeze(1)  # [batch, emb_dim]
            output = output.squeeze(1)
            context = context.squeeze(1)
            prediction = self.fc_out(torch.cat((output, context, embedded ), dim=1))
            # concat on feature dimension: [batch, hid_dim * 2 + emb_dim]
            # linear maps to vocab size:  [batch, hid_dim * 2 + emb_dim]    → [batch, output_dim]
            
            return prediction, hidden, cell, attn_weights
            # prediction: [batch, output_dim] — scores for each token in the output vocabulary
            # hidden, cell: updated memory for next step
            # attn_weights: attention weights for visualization

        else:

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
    def __init__(self, encoder, decoder, dataset, pad_idx, attention=None, bidirectional=None):
        super().__init__()
        self.encoder   = encoder
        self.decoder   = decoder
        self.dataset   = dataset
        self.pad_idx   = pad_idx
        self.use_attention = attention is not None
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        
        # self.init_weights() 

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size     = src.shape[0]
        trg_len        = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs  = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device) 
        if self.use_attention:
            encoder_outputs, hidden, cell = self.encoder(src)
            x = trg[:, 0]

            for t in range(1, trg_len):
                output, hidden, cell, _ = self.decoder(x, hidden, cell, encoder_outputs)
                outputs[:, t, :] = output
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                x = trg[:, t] if teacher_force else output.argmax(dim=1)

            return outputs
        else:
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
        end_idx    = self.dataset.sql_vocab.stoi["<end>"]
        finished   = torch.zeros(batch_size, dtype=torch.bool).to(src.device)
        outputs    = []

        with torch.no_grad():
            if self.use_attention:
                encoder_outputs, hidden, cell = self.encoder(src)
            else:
                encoder_outputs = None
                hidden, cell = self.encoder(src)

            x = torch.full(
                (batch_size,),
                fill_value=self.dataset.sql_vocab.stoi["<start>"],
                dtype=torch.long,
                device=src.device
            )
            for _ in range(max_len):
                # FIX: was hardcoded to non-attention path
                if self.use_attention:
                    output, hidden, cell, _ = self.decoder(x, hidden, cell, encoder_outputs)
                else:
                    output, hidden, cell = self.decoder(x, hidden, cell)
                x = output.argmax(dim=1)
                outputs.append(x.clone())
                finished |= (x == end_idx)
                if finished.all():
                    break

        preds = torch.stack(outputs, dim=1)             # [batch, seq_len]
        return [self.dataset.sql_vocab.decode(p) for p in preds]
    

    def translate(self, sentence: str, schema=None, max_len=50):
        was_training = self.training
        self.eval()

        if schema:
            parsed   = parse_schema(schema)
            sentence = f"{sentence} | {parsed}"

        tokens  = self.dataset.text_vocab.encode(sentence).unsqueeze(0).to(self.device)
        end_idx = self.dataset.sql_vocab.stoi["<end>"]
        result  = []

        with torch.no_grad():
            if self.use_attention:
                encoder_outputs, hidden, cell = self.encoder(tokens)
            else:
                encoder_outputs = None
                hidden, cell    = self.encoder(tokens)

            x = torch.tensor([self.dataset.sql_vocab.stoi["<start>"]], device=self.device)

            for _ in range(max_len):
                if self.use_attention:
                    output, hidden, cell, _ = self.decoder(x, hidden, cell, encoder_outputs)
                else:
                    output, hidden, cell    = self.decoder(x, hidden, cell)

                x     = output.argmax(dim=1)
                token = self.dataset.sql_vocab.itos[x.item()]
                if token == "<end>":
                    break
                result.append(token)

        # ← outside the with block
        if was_training:
            self.train()
        return " ".join(result)
    
input_dim = len(text_vocab)  # source vocabulary size
output_dim = len(sql_vocab)  # target vocabulary size

enc = Encoder(input_dim, enc_emb_dim, hid_dim, num_layers, dropout, attention=attention, bidirectional=bidirectional).to(device)
dec = Decoder(output_dim, dec_emb_dim, hid_dim, num_layers, dropout, attention=attention, bidirectional=bidirectional).to(device)
model = Seq2Seq(enc, dec, dataset, pad_idx, attention=attention, bidirectional=bidirectional).to(device)