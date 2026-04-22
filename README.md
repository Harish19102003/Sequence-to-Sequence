# Sequence to Sequence

This project implements **Sequence to Sequence(Seq2Seq)** using **PyTorch Lightning** to generate **SQL QUERY**.

##  Project Structure

```
TEXT-to-SQL Database Query/
├── data                   # Data
├── README.md              # Documentation
├── requirements.txt       # Dependencies
├── .gitignore             # Ignore cache/checkpoint/log files
├── checkpoints/           # Model checkpoints
├── config.py              # Model Hyperparameters
├── dataset.py             # Dataset loading
├── model.py               # seq2seq model 
├── train.py               # Main training script
├── utils.py               # Model loading and evaluation
```

---

##  Usage

### Install Dependencies
```bash
git clone https://github.com/Harish19102003/Sequence-to-Sequence.git
cd Sequence-to-Sequence
pip install -r requirements.txt
```

##  Dataset

- **Name:** Anime Face Dataset  
- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/meryentr/text-to-sql)  
- **Description:** This dataset consists of 65354 entries designed to evaluate the performance of text-to-SQL models. Each entry contains a natural language text query and its corresponding SQL command. 

---

## Download Dataset
```bash
kaggle datasets download -d meryentr/text-to-sql
unzip text-to-sql.zip -d data/
```

## Train :
```bash
python -m train.py
```

## Evaluation
Using BLEU Score
```bash
python -m utils.py
```

## TensorBoard
```bash
tensorboard --logdir tb_logs
```

