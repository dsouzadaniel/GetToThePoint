from typing import List
import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo, batch_to_ids

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path().resolve()))

# Project Imports
from data import loader
from utils import constant, helper


class PointerGenerator(nn.Module):
    def __init__(self,
                 vocab: List,
                 elmo_weights_file: str,
                 elmo_options_file: str,
                 elmo_embed_dim: int,
                 elmo_sent: bool = False,
                 alignment_model: str = "additive"):
        super(PointerGenerator, self).__init__()

        # Model Properties
        self.elmo_sent = elmo_sent
        self.alignment_model = alignment_model
        self.randomize_init_hidden = True
        self.vocab = sorted(vocab)
        self.vocab_2_ix = {k: v for k, v in zip(self.vocab, range(0, len(self.vocab)))}
        self.ix_2_vocab = {v: k for k, v in self.vocab_2_ix.items()}

        self.map_vocab_2_ix = lambda p_t: [[self.vocab_2_ix[w_t] for w_t in s_t] for s_t in p_t]
        self.map_ix_2_vocab = lambda p_i: [[self.ix_2_vocab[w_i] for w_i in s_i] for s_i in p_i]

        # Model Constants

        self.ELMO_EMBED_DIM = elmo_embed_dim  # This will change if the ELMO options/weights change

        self.VOCAB_SIZE = len(self.vocab)

        # Model Layers
        self.elmo = Elmo(elmo_options_file, elmo_weights_file, 1)
        self.encoder = nn.LSTM(input_size=self.ELMO_EMBED_DIM,
                               hidden_size=self.ELMO_EMBED_DIM,
                               num_layers=1,
                               bidirectional=True)

        self.decoder = nn.LSTM(input_size=self.ELMO_EMBED_DIM,
                               hidden_size=2 * self.ELMO_EMBED_DIM,
                               num_layers=1,
                               bidirectional=False)

        self.Wh = nn.Linear(in_features=2 * self.ELMO_EMBED_DIM,
                            out_features=2 * self.ELMO_EMBED_DIM,
                            bias=False)
        self.Ws = nn.Linear(in_features=2 * self.ELMO_EMBED_DIM,
                            out_features=2 * self.ELMO_EMBED_DIM,
                            bias=True)
        self.v = nn.Linear(in_features=2 * self.ELMO_EMBED_DIM,
                           out_features=1,
                           bias=False)

        self.sm_dim0 = nn.Softmax(dim=0)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.Vocab_Project_1 = nn.Linear(in_features=4 * self.ELMO_EMBED_DIM,
                                         out_features=8 * self.ELMO_EMBED_DIM,
                                         bias=True)

        self.Vocab_Project_2 = nn.Linear(in_features=8 * self.ELMO_EMBED_DIM,
                                         out_features=self.VOCAB_SIZE,
                                         bias=True)

        self.Wh_pgen = nn.Linear(in_features=2 * self.ELMO_EMBED_DIM, out_features=1, bias=False)
        self.Ws_pgen = nn.Linear(in_features=2 * self.ELMO_EMBED_DIM, out_features=1, bias=False)
        self.Wx_pgen = nn.Linear(in_features=self.ELMO_EMBED_DIM, out_features=1, bias=True)

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

    def _embed_doc(self, doc_tokens: List[List[str]], **kwargs) -> torch.Tensor:
        # Embed the Doc with Elmo
        doc_embedded_elmo = self._elmo_embed_doc(doc_tokens)

        print("Pre Doc Shape -> {0}".format(doc_embedded_elmo.shape))

        prepend = kwargs.get('prepend_START', None)
        if prepend:
            start_token_elmo = self._elmo_embed_doc([['<START>']])
            doc_embedded_elmo = torch.cat((start_token_elmo, doc_embedded_elmo[:-1]), dim=0)

        print("Post Doc Shape -> {0}".format(doc_embedded_elmo.shape))

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
        if alignment_model == "additive":
            # Attention Alignment Model from Bahdanau et al(2015)
            e = self.v(self.tanh(self.Wh(h) + self.Ws(s))).squeeze()
        elif alignment_model == "dot_product":
            # Attention Alignment Model from Luong et al(2015)
            e = torch.matmul(h, s.squeeze(dim=0))
        return e

    def _decoder_train(self, encoder_hidden_states, src_tokens, tgt_tokens):
        _init_probe = encoder_hidden_states[-1].reshape(1, 1, -1)
        curr_h, curr_c = (_init_probe, torch.randn_like(_init_probe))

        tgt_elmo = self._embed_doc(tgt_tokens, prepend_START=True)

        flat_src_tokens = [i for j in src_tokens for i in j]
        flat_tgt_tokens = [i for j in tgt_tokens for i in j]

        assert len(flat_src_tokens) == encoder_hidden_states.shape[0]

        print(flat_src_tokens[:5])

        new_words = sorted([w for w in flat_src_tokens if w not in self.vocab])

        extended_vocab = self.vocab + new_words
        extended_vocab_2_ix = {**self.vocab_2_ix, **{w: ix for w, ix in zip(new_words, range(
            len(self.vocab), len(extended_vocab)))}}
        extended_ix_2_vocab = {v: k for k, v in extended_vocab_2_ix.items()}

        assert len(extended_vocab) == len(extended_vocab_2_ix) == len(extended_ix_2_vocab)

        # [1:] to align because of the <START> token
        gold_vocab_ixs = torch.LongTensor([extended_vocab_2_ix[w] for w in flat_tgt_tokens[1:]])

        print("NEW WORDS ->{0}".format(new_words))

        collected_summary = []

        # To calculate loss
        collect_xtnd_vocab_prjtns = []

        for curr_elmo in tgt_elmo:
            p_vocab = torch.zeros(size=(1, len(extended_vocab)))
            p_attn = torch.zeros(size=(1, len(extended_vocab)))

            curr_i = curr_elmo.reshape(1, 1, -1)

            curr_o, (curr_h, curr_c) = self.decoder(curr_i, (curr_h, curr_c))

            # Calculate Context Vector
            curr_attn = self._align(s=curr_h.squeeze(dim=1), h=encoder_hidden_states,
                                    alignment_model=self.alignment_model)
            curr_attn = self.sm_dim0(curr_attn)
            curr_ctxt = torch.matmul(curr_attn, encoder_hidden_states)

            # Concatenate Context & Decoder Hidden State
            state_ctxt_concat = torch.cat((curr_h.squeeze(), curr_ctxt))

            # Project to Vocabulary
            vocab_prjtn = self.Vocab_Project_2(self.Vocab_Project_1(state_ctxt_concat))

            # print("ATTN ->{0}".format(curr_attn.shape))
            # print("INTO ->{0}".format(vocab_prjtn.shape))

            p_vocab[:, :self.VOCAB_SIZE] = vocab_prjtn
            for src_word, src_attn in zip(flat_src_tokens, curr_attn):
                p_attn[:, extended_vocab_2_ix[src_word]] += src_attn

            p_gen = self.sigmoid(
                self.Wh_pgen(curr_ctxt) + self.Ws_pgen(curr_h.squeeze()) + self.Wx_pgen(curr_i.squeeze()))

            # x_1 = p_vocab
            # x_2 = p_gen * p_vocab + (1 - p_gen) * p_vocab

            # print(x_1[:, :10])
            # print(x_2[:, :10])
            #
            # print(x_1[:, -10:])
            # print(x_2[:, -10:])

            # print("EQUAL -> {0}".format(torch.equal(input=x_1, other=x_2)))

            # print("ATTN SHAPE ->{0}".format(curr_attn.shape))
            # print("WORD SHAPE ->{0}".format(len(flat_src_tokens)))
            # print(p_vocab)

            # print("VOCAB_PROJECTION ->", vocab_projection.shape)
            predicted_vocab_ix = vocab_projection.argmax(dim=0).item()
            predicted_word = self.ix_2_vocab[predicted_vocab_ix]
            # print("PREDICTED_WORD ->", predicted_word)
            collected_summary.append(predicted_word)

            collect_xtnd_vocab_prjtns.append()

        predicted_vocab_projections = torch.stack(predicted_vocab_projections, dim=0)

        print("X ->{0}".format(predicted_vocab_projections.shape))
        print("Y ->{0}".format(gold_vocab_ixs.shape))

        print("GOLD_SUMMARY ->\n{0}".format(' '.join(target_summary)))
        print("GENERATED_SUMMARY->\n{0}".format(' '.join(collected_summary)))
        return

    def _decoder_test(self, encoder_hidden_states, len_of_summary: int = 20):
        _init_probe = encoder_hidden_states[-1].reshape(1, 1, -1)
        curr_h, curr_c = (_init_probe, torch.randn_like(_init_probe))

        curr_attn = self._align(s=curr_h.squeeze(dim=1), h=encoder_hidden_states, alignment_model=self.alignment_model)
        curr_attn = self.sm_dim0(curr_attn)
        curr_ctxt = torch.matmul(curr_attn, encoder_hidden_states)

        collected_summary_tokens = [['<START>']]

        curr_elmo = self._embed_doc(doc_tokens=collected_summary_tokens)

        for token_ix in range(len_of_summary):
            curr_i = torch.cat((curr_ctxt, curr_elmo), dim=0).reshape(1, 1, -1)

            curr_o, (curr_h, curr_c) = self.decoder(curr_i, (curr_h, curr_c))

            curr_attn = self._align(s=curr_h.squeeze(dim=1), h=encoder_hidden_states,
                                    alignment_model=self.alignment_model)
            curr_attn = self.sm_dim0(curr_attn)
            curr_ctxt = torch.matmul(curr_attn, encoder_hidden_states)

            # Output
            state_ctxt_concat = torch.cat((curr_h.squeeze(), curr_ctxt))

            vocab_projection = self.Vocab_Project_2(self.Vocab_Project_1(state_ctxt_concat))

            curr_pred_token = self.ix_2_vocab[vocab_projection.argmax().item()]

            collected_summary_tokens[-1][-1].append(curr_pred_token)

            # Just get the Elmo Embedding of the Current Sentence
            curr_elmo = self._embed_doc([collected_summary_tokens[-1]])

            if curr_pred_token == '.':
                # Start a New Line
                collected_summary_tokens.append([])

        return collected_summary_tokens

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
            loss = self._decoder_train(encoder_states, orig_text_tokens, summ_text_tokens)
        else:
            # -> Inference Loop
            print("Testing")
            pred_summ_text_tokens = self._decoder_test(encoder_states, len_of_summary=30)
            pass

        return None


init_vocab = []
with open('init_vocab_str.txt') as f:
    for line in f.readlines():
        init_vocab.append(line.strip())

model = PointerGenerator(vocab=init_vocab,
                         alignment_model="additive",
                         elmo_embed_dim=constant.ELMO_EMBED_DIM,
                         elmo_weights_file=constant.ELMO_WEIGHTS_FILE,
                         elmo_options_file=constant.ELMO_OPTIONS_FILE)

input_texts = ["Hello World. This is a great world. I love NLP!"]
output_texts = ["Hello World! I love NLP"]

input_text_tokens = [helper.tokenize_en(input_text, lowercase=True) for input_text in input_texts]
output_text_tokens = [helper.tokenize_en(output_text, lowercase=True) for output_text in output_texts]

tensor = model(orig_text_tokens=input_text_tokens[0], summ_text_tokens=output_text_tokens[0])
# print("Output Tensor Shape is :{0}".format(tensor.shape))
