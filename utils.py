import torch
from pathlib import Path
import os
from torchtext.data.metrics import bleu_score
from model import Seq2Seq, Encoder, Decoder
from config import  INPUT_DIM, ENC_EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT, OUTPUT_DIM, DEC_EMB_DIM, sql_vocab


def load_model(output_file):

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(enc, dec)
    checkpoint = torch.load(output_file)
    
    # Lightning wraps weights inside "state_dict" key
    state_dict = checkpoint["state_dict"]
    
    model.load_state_dict(state_dict)
    return model.eval()

def main():
    model = load_model(output_file)

    question = "What are the different schools and their nicknames ordered by their founding years"
    ref = "SELECT school_name, nickname FROM schools ORDER BY founding_year"

    reference  = sql_vocab.tokenizer(ref)
    predicted  = sql_vocab.tokenizer(model.translate(question))

    print("Question:  ", question)
    print("Reference: ", reference)
    print("Predicted: ", predicted)

    score = bleu_score([predicted], [[reference]])
    print("BLEU:      ", score)

if __name__ == "__main__":
    output_file="checkpoints/text_to_sql.ckpt"
    if os.path.exists(output_file):
         main()
    else:
         print("There no model trained")