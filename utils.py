import torch
from pathlib import Path
import os
from torchtext.data.metrics import bleu_score
from model import model, device
from train import trainer
from dataset import test_loader, sql_vocab
import warnings
warnings.filterwarnings("ignore")

def load_model(output_file, model = model):
    
    checkpoint = torch.load(output_file)
    
    # Lightning wraps weights inside "state_dict" key
    state_dict = checkpoint["state_dict"]
    
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def main():
    
    output_file = Path("checkpoints/text_to_sql.ckpt")
    
    if not os.path.exists(output_file):
        print("There is no model trained")
        
    else:
        model = load_model(output_file).eval()
        pred = trainer.predict(model, test_loader)
        pred = [seq for batch in pred for seq in batch]  # type: ignore
        pred_tokens = [seq.split() for seq in pred]
        ref_tokens  = [sql_vocab.decode(ref).split() for ref, _ in test_loader.dataset]
        score = bleu_score(pred_tokens, [[ref] for ref in ref_tokens])
        print("BLEU:      ", score)

if __name__ == "__main__": 
    main()
