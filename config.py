# Model Hyperparameters :
enc_emb_dim      = 512              # encoder word embedding dimensionality
dec_emb_dim      = 512              # decoder word embedding dimensionality
hid_dim          = 256              # cells per layer
num_layers       = 2                # layer deep lstm
dropout          = 0.5              # dropout probability       
epochs           = 20          
grad_clip        = 1.0
batch_size       = 32

attention        = True
bidirectional    = True