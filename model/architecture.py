from typing import List
import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo, batch_to_ids

import spacy

nlp = spacy.load('en_core_web_sm')


class BaselineSeq2Seq2wAttn(nn.Module):
    def __init__(self, elmo_sent: bool = False):
        super(BaselineSeq2Seq2wAttn, self).__init__()

        # Model Properties
        self.elmo_sent = elmo_sent


        # Model Constants
        self.ELMO_EMBED_DIM = 256  # This will change if the ELMO options/weights change
        self.WEIGHTS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
        self.OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"

        # Model Layers
        self.elmo = Elmo(self.OPTIONS_FILE, self.WEIGHTS_FILE, 1)

    def _elmo_embed_doc(self, doc_tokens: List[List[str]]) -> torch.Tensor:
        if not self.elmo_sent:
            doc_tokens = [[token for sent_tokens in doc_tokens for token in sent_tokens]]

        doc_elmo_ids = batch_to_ids(doc_tokens)
        doc_elmo_embed = self.elmo(doc_elmo_ids)

        if self.elmo_sent:
            _elmo_doc_feats = []
            for sent_elmo_embed, sent_elmo_mask in zip(doc_elmo_embed['elmo_representations'][0],
                                                       doc_elmo_embed['mask']):
                _elmo_doc_feats.append(sent_elmo_embed[:sum(sent_elmo_mask)])
            elmo_doc_feats = torch.cat(_elmo_doc_feats, dim=0)
        else:
            elmo_doc_feats = doc_elmo_embed['elmo_representations'][0][0]
        return elmo_doc_feats

    def _embed_doc(self, doc: str) -> torch.Tensor:
        doc = nlp(doc)
        doc_tokens = [[token.text for token in sent] for sent in doc.sents]
        # Embed the Doc with Elmo
        doc_embedded_elmo = self._elmo_embed_doc(doc_tokens)
        return doc_embedded_elmo

    def forward(self, x:str) -> torch.Tensor:
        return self._embed_doc(x)


model = BaselineSeq2Seq2wAttn()

input_text = "Hello World. This is great"

output_tensor = model(input_text)

print("Output Tensor Shape is :{0}".format(output_tensor.shape))
