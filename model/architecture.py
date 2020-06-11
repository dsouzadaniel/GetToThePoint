from typing import List
import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo, batch_to_ids

import spacy

nlp = spacy.load('en_core_web_sm')


class BaselineSeq2Seq2wAttn(nn.Module):
    def __init__(self, elmo_sent: bool = False, alignment_model: str = "dot_product"):
        super(BaselineSeq2Seq2wAttn, self).__init__()

        # Model Properties
        self.elmo_sent = elmo_sent
        self.alignment_model = alignment_model
        self.randomize_init_hidden = True

        # Model Constants

        self.ELMO_EMBED_DIM = 256  # This will change if the ELMO options/weights change
        self.WEIGHTS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
        self.OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"

        self.VOCAB_SIZE = 16*self.ELMO_EMBED_DIM

        # Model Layers
        self.elmo = Elmo(self.OPTIONS_FILE, self.WEIGHTS_FILE, 1)
        self.encoder = nn.LSTM(input_size=self.ELMO_EMBED_DIM,
                               hidden_size=self.ELMO_EMBED_DIM,
                               num_layers=1,
                               bidirectional=True)

        self.decoder = nn.LSTM(input_size=3 * self.ELMO_EMBED_DIM,
                               hidden_size=2 * self.ELMO_EMBED_DIM,
                               num_layers=1,
                               bidirectional=False)

        self.Wh = nn.Linear(in_features=2*self.ELMO_EMBED_DIM,
                            out_features=2*self.ELMO_EMBED_DIM,
                            bias=False)
        self.Ws = nn.Linear(in_features=2*self.ELMO_EMBED_DIM,
                            out_features=2*self.ELMO_EMBED_DIM,
                            bias=True)
        self.v = nn.Linear(in_features=2*self.ELMO_EMBED_DIM,
                           out_features=1,
                           bias=False)

        self.sm_dim0 = nn.Softmax(dim=0)
        self.tanh = nn.Tanh()

        self.Vocab_Project_1 = nn.Linear(in_features=4*self.ELMO_EMBED_DIM,
                                         out_features=8*self.ELMO_EMBED_DIM,
                                         bias=True)

        self.Vocab_Project_2 = nn.Linear(in_features=8*self.ELMO_EMBED_DIM,
                                         out_features=self.VOCAB_SIZE,
                                         bias=True)

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

    def _embed_doc(self, doc_tokens: str, **kwargs) -> torch.Tensor:
        prepend = kwargs.get('prepend', None)
        if prepend:
            doc_tokens[0] = prepend + doc_tokens[0]
        print(doc_tokens)
        # Embed the Doc with Elmo
        doc_embedded_elmo = self._elmo_embed_doc(doc_tokens)
        return doc_embedded_elmo

    def _init_bi_hidden(self, batch_size: int = 1, num_layers: int = 1):
        if self.randomize_init_hidden:
            init_hidden = torch.randn(num_layers * 2, batch_size,
                                      self.ELMO_EMBED_DIM)
        else:
            init_hidden = torch.zeros(num_layers * 2, batch_size,
                                      self.ELMO_EMBED_DIM)
        return init_hidden, init_hidden

    def _run_through_bilstm(self, input_tensor: torch.Tensor, bilstm: torch.nn.modules.rnn):
        init_bi_hidden = self._init_bi_hidden(num_layers=bilstm.num_layers)
        output_tensor, _ = bilstm(input_tensor, init_bi_hidden)
        output_tensor = output_tensor.view(input_tensor.shape[0], 1, 2, bilstm.hidden_size)
        output_tensor = torch.cat((output_tensor[:, :, 0, :],
                                   output_tensor[:, :, 1, :]),
                                  dim=2).squeeze(dim=1)
        return output_tensor

    def _align(self, s, h, alignment_model="additive"):
        if alignment_model =="additive":
            # Attention Alignment Model from Bahdanau et al(2015)
            e = self.v(self.tanh(self.Wh(h)+self.Ws(s))).squeeze()
        elif alignment_model == "dot_product":
            # Attention Alignment Model from Luong et al(2015)
            e = torch.matmul(h, s.squeeze(dim=0))
        return e

    def _decoder_train(self, encoder_hidden_states, target_tokens):
        _init_probe = encoder_hidden_states[-1].reshape(1, 1, -1)
        curr_h, curr_c = (_init_probe, torch.randn_like(_init_probe))

        curr_attn = self._align(s=curr_h.squeeze(dim=1), h=encoder_hidden_states, alignment_model=self.alignment_model)
        curr_attn = self.sm_dim0(curr_attn)
        curr_ctxt = torch.matmul(curr_attn, encoder_hidden_states)

        target_elmo = self._embed_doc(target_tokens, prepend=["<START>"])

        for curr_elmo in target_elmo:
            curr_i = torch.cat((curr_ctxt, curr_elmo), dim=0).reshape(1, 1, -1)

            curr_o, (curr_h, curr_c) = self.decoder(curr_i, (curr_h, curr_c))

            curr_attn = self._align(s=curr_h.squeeze(dim=1), h=encoder_hidden_states, alignment_model=self.alignment_model)
            curr_attn = self.sm_dim0(curr_attn)
            curr_ctxt = torch.matmul(curr_attn, encoder_hidden_states)

            # Output
            state_ctxt_concat = torch.cat((curr_h.squeeze(), curr_ctxt))

            vocab_projection = self.Vocab_Project_2(self.Vocab_Project_1(state_ctxt_concat))

            print("VOCAB_PROJECTION ->",vocab_projection.shape)

        print("Golly! ^_^ ")
        return

    def forward(self, orig_text_tokens: List[List[str]], **kwargs) -> torch.Tensor:
        # Embed the Orig with Elmo
        orig_embedded_elmo = self._embed_doc(orig_text_tokens)

        # Encode with BiLSTM
        orig_embedded_elmo.unsqueeze_(dim=1)
        encoder_states = self._run_through_bilstm(orig_embedded_elmo, self.encoder)

        # summ_text implies training
        summ_text_tokens = kwargs.get('summ_text_tokens', None)

        if summ_text_tokens:
            # -> Training Loop
            print("Training")
            loss = self._decoder_train(encoder_states, summ_text_tokens)
        else:
            # -> Inference Loop
            print("Testing")
            pass

        return encoder_states


model = BaselineSeq2Seq2wAttn(alignment_model="additive")

def tokenize_en(text:str):
    doc = nlp(text)
    doc_tokens = [[token.text for token in sent] for sent in doc.sents]
    return doc_tokens

input_texts = ["Hello World. This is great. I love NLP!"]
output_texts = ["Hey great world! I love NLP"]

input_text_tokens = [tokenize_en(input_text) for input_text in input_texts]
output_text_tokens = [tokenize_en(output_text) for output_text in output_texts]


# tensor = model(orig_text=input_text)
# print("Output Tensor Shape is :{0}".format(tensor.shape))

tensor = model(orig_text_tokens=input_text_tokens[0], summ_text_tokens=output_text_tokens[0])
# print("Output Tensor Shape is :{0}".format(tensor.shape))
