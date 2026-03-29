# TEXT-to-SQL Database Query

This project implements **Sequence to Sequence(Seq2Seq)** using **PyTorch Lightning** to generate **SQL QUERY**.

##  Project Structure

```
TEXT-to-SQL Database Query/
├── data                   # Data
├── README.md              # Documentation
├── requirements.txt       # Dependencies
├── .gitignore             # Ignore cache/checkpoint/log files
├── checkpoints/           # Model checkpoints
├── config.py              
├── dataset.py             
├── model.py               # seq2seq model 
├── train.py               # Main training script
├── utils.py               
```

---

##  Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Train :
```bash
python -m train.py
```

## Evaluation
Using BLEU Score

### Generate SQL Query:
```bash
python -m utils.py
```

