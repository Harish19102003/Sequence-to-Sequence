from dataset import Build_Dataset, Vocabulary


root_dir = "data/spider-text-sql/spider_text_sql.csv"

dataset = Build_Dataset(root_dir, Vocabulary)
text_vocab = dataset.text_vocab
sql_vocab  = dataset.sql_vocab

# Model Hyperparameters :
INPUT_DIM = len(text_vocab)  # Source vocabulary size
OUTPUT_DIM = len(sql_vocab)  # Target vocabulary size
ENC_EMB_DIM = 1000  # Word embedding dimensionality
DEC_EMB_DIM = 1000
HID_DIM = 1000      # 1000 cells per layer
NUM_LAYERS = 4        # 4-layer deep LSTM
DROPOUT = 0.6       
epochs = 50
grad_clip = 1.0
batch_size = 32