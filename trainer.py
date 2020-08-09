from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from gttp_lightning import PointerGenerator

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve()))

from utils import constant

init_vocab = []
with open('init_vocab_str.txt') as f:
    for line in f.readlines():
        init_vocab.append(line.strip())

assert init_vocab[constant.UNK_TOK_IX] == "<UNK>", "<UNK> Token not found at 0 position "

model = PointerGenerator(vocab=init_vocab,
                         elmo_embed_dim=constant.ELMO_EMBED_DIM,
                         elmo_weights_file=constant.ELMO_WEIGHTS_FILE,
                         elmo_options_file=constant.ELMO_OPTIONS_FILE)


checkpoint_GTTP = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=1,
    verbose=True,
    monitor='loss/validation',
    mode='min',
    prefix=''
)

trainer = Trainer(min_epochs=5, checkpoint_callback=checkpoint_GTTP)
trainer.fit(model)