from typing import List, Union, Tuple, Dict
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

loss_fn = nn.CrossEntropyLoss()

from pytorch_lightning.core.lightning import LightningModule


class PointerGenerator(LightningModule):
    def __init__(self,
                 vocab: List,
                 elmo_weights_file: str,
                 elmo_options_file: str,
                 elmo_embed_dim: int,
                 elmo_sent: bool = False,
                 alignment_model: str = "additive"):
        super().__init__()

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
        self.elmo.eval()

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
        #
        # print("Pre Doc Shape -> {0}".format(doc_embedded_elmo.shape))

        prepend = kwargs.get('prepend_START', None)
        if prepend:
            start_token_elmo = self._elmo_embed_doc([['<START>']])
            doc_embedded_elmo = torch.cat((start_token_elmo, doc_embedded_elmo[:-1]), dim=0)

        # print("Post Doc Shape -> {0}".format(doc_embedded_elmo.shape))

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

    def _decoder_train(self, encoder_states, src_tokens, tgt_tokens):
        _init_probe = encoder_states[-1].reshape(1, 1, -1)
        curr_h, curr_c = (_init_probe, torch.randn_like(_init_probe))

        flat_src_tokens = [i for j in src_tokens for i in j]

        assert len(flat_src_tokens) == encoder_states.shape[0]

        new_words = sorted(list(set([w for w in flat_src_tokens if w not in self.vocab])))

        extended_vocab = self.vocab + new_words
        _extend_2_ix = {w: ix for w, ix in zip(new_words, range(len(self.vocab), len(extended_vocab)))}
        extended_vocab_2_ix = {**self.vocab_2_ix, **_extend_2_ix}
        extended_ix_2_vocab = {v: k for k, v in extended_vocab_2_ix.items()}

        assert len(extended_vocab) == len(extended_vocab_2_ix) == len(extended_ix_2_vocab)

        # To calculate loss
        collect_xtnd_vocab_prjtns = []

        tgt_elmo = self._embed_doc(tgt_tokens, prepend_START=True)
        for curr_elmo in tgt_elmo:
            p_vocab = torch.zeros(size=(1, len(extended_vocab)))
            p_attn = torch.zeros(size=(1, len(extended_vocab)))

            curr_i = curr_elmo.reshape(1, 1, -1)
            curr_o, (curr_h, curr_c) = self.decoder(curr_i, (curr_h, curr_c))

            # Calculate Context Vector
            curr_attn = self._align(s=curr_h.squeeze(dim=1), h=encoder_states,
                                    alignment_model=self.alignment_model)
            curr_attn = self.sm_dim0(curr_attn)
            curr_ctxt = torch.matmul(curr_attn, encoder_states)

            # Concatenate Context & Decoder Hidden State
            state_ctxt_concat = torch.cat((curr_h.squeeze(), curr_ctxt))

            # Project to Vocabulary
            vocab_prjtn = self.Vocab_Project_2(self.Vocab_Project_1(state_ctxt_concat))
            p_vocab[:, :self.VOCAB_SIZE] = vocab_prjtn
            for src_word, src_attn in zip(flat_src_tokens, curr_attn):
                p_attn[:, extended_vocab_2_ix[src_word]] += src_attn

            p_gen = self.sigmoid(
                self.Wh_pgen(curr_ctxt) + self.Ws_pgen(curr_h.squeeze()) + self.Wx_pgen(curr_i.squeeze()))

            p_W = p_gen * p_vocab + (1 - p_gen) * p_attn
            collect_xtnd_vocab_prjtns.append(p_W)

        collect_xtnd_vocab_prjtns = torch.cat(collect_xtnd_vocab_prjtns, dim=0)

        return (collect_xtnd_vocab_prjtns, extended_vocab_2_ix, extended_ix_2_vocab)

    def _decoder_test(self, encoder_states, src_tokens, len_of_summary: int):
        _init_probe = encoder_states[-1].reshape(1, 1, -1)
        curr_h, curr_c = (_init_probe, torch.randn_like(_init_probe))

        flat_src_tokens = [i for j in src_tokens for i in j]

        assert len(flat_src_tokens) == encoder_states.shape[0]

        new_words = sorted(list(set([w for w in flat_src_tokens if w not in self.vocab])))

        extended_vocab = self.vocab + new_words
        _extend_2_ix = {w: ix for w, ix in zip(new_words, range(len(self.vocab), len(extended_vocab)))}
        extended_vocab_2_ix = {**self.vocab_2_ix, **_extend_2_ix}
        extended_ix_2_vocab = {v: k for k, v in extended_vocab_2_ix.items()}

        assert len(extended_vocab) == len(extended_vocab_2_ix) == len(extended_ix_2_vocab)

        collect_xtnd_vocab_prjtns = []

        # for curr_elmo in tgt_elmo:
        collected_summary_tokens = [['<START>']]
        curr_elmo = self._embed_doc(doc_tokens=collected_summary_tokens)

        for token_ix in range(len_of_summary):
            p_vocab = torch.zeros(size=(1, len(extended_vocab)))
            p_attn = torch.zeros(size=(1, len(extended_vocab)))

            curr_i = curr_elmo.reshape(1, 1, -1)
            curr_o, (curr_h, curr_c) = self.decoder(curr_i, (curr_h, curr_c))

            # Calculate Context Vector
            curr_attn = self._align(s=curr_h.squeeze(dim=1), h=encoder_states,
                                    alignment_model=self.alignment_model)
            curr_attn = self.sm_dim0(curr_attn)
            curr_ctxt = torch.matmul(curr_attn, encoder_states)

            # Concatenate Context & Decoder Hidden State
            state_ctxt_concat = torch.cat((curr_h.squeeze(), curr_ctxt))

            # Project to Vocabulary
            vocab_prjtn = self.Vocab_Project_2(self.Vocab_Project_1(state_ctxt_concat))
            p_vocab[:, :self.VOCAB_SIZE] = vocab_prjtn
            for src_word, src_attn in zip(flat_src_tokens, curr_attn):
                p_attn[:, extended_vocab_2_ix[src_word]] += src_attn

            p_gen = self.sigmoid(
                self.Wh_pgen(curr_ctxt) + self.Ws_pgen(curr_h.squeeze()) + self.Wx_pgen(curr_i.squeeze()))

            p_W = p_gen * p_vocab + (1 - p_gen) * p_attn
            collect_xtnd_vocab_prjtns.append(p_W)

            curr_pred_token = extended_ix_2_vocab[p_W.argmax(dim=1).item()]
            collected_summary_tokens[-1].append(curr_pred_token)

            # Just get the Elmo Embedding of the Latest Word of the Latest Sentence
            curr_elmo = self._embed_doc([collected_summary_tokens[-1]])[-1]

            if curr_pred_token == '.':
                # Start a New Line
                collected_summary_tokens.append([])

        collect_xtnd_vocab_prjtns = torch.cat(collect_xtnd_vocab_prjtns, dim=0)

        return (collect_xtnd_vocab_prjtns, extended_vocab_2_ix, extended_ix_2_vocab)

    def _extend_vocab(self, possible_new_tokens):
        new_words = sorted(list(set([w for w in possible_new_tokens if w not in self.vocab])))
        extended_vocab = self.vocab + new_words
        extended_vocab_2_ix = {**self.vocab_2_ix, **{w: ix for w, ix in zip(new_words, range(
            len(self.vocab), len(extended_vocab)))}}
        extended_ix_2_vocab = {v: k for k, v in extended_vocab_2_ix.items()}

        assert len(extended_vocab) == len(extended_vocab_2_ix) == len(extended_ix_2_vocab), "Vocab Length Mismatch"

        return extended_vocab, extended_vocab_2_ix, extended_ix_2_vocab

    def training_step(self, batch):
        batch_loss = None
        for orig_text, summ_text in batch:

    def forward(self, orig_text: str, **kwargs) -> Union:
        orig_tokens = helper.tokenize_en(orig_text, lowercase=True)

        # Extend the vocabulary to include new words
        orig_tokens_flat = [i for j in orig_tokens for i in j]
        ex_vocab, ex_vocab_2_ix, ex_ix_2_vocab = self._extend_vocab(possible_new_tokens=orig_tokens_flat)

        # Embed the Orig with Elmo
        orig_elmo = self._embed_doc(orig_tokens)
        # Encode with BiLSTM
        orig_elmo.unsqueeze_(dim=1)
        encoder_states = self._run_through_bilstm(orig_elmo, self.encoder)
        assert len(orig_tokens_flat) == encoder_states.shape[0]

        # summ_text implies training
        summ_text = kwargs.get('summ_text', None)

        if summ_text:
            summ_tokens = helper.tokenize_en(summ_text, lowercase=True)
            summ_elmo = self._embed_doc(summ_tokens, prepend_START=True)
            summ_len = len(summ_elmo)
        else:
            summ_len = kwargs.get('summ_len', None)
            generated_summ_tokens = [['<START>']]

        # To calculate loss
        vocab_prjtns = []
        _init_probe = encoder_states[-1].reshape(1, 1, -1)
        curr_deco_state = (_init_probe, torch.randn_like(_init_probe))
        curr_pred_token = None
        for token_ix in range(summ_len):
            if summ_text:
                curr_i = summ_elmo[token_ix].reshape(1, 1, -1)
            elif curr_pred_token:
                # Append currently predicted token
                generated_summ_tokens[-1].append(curr_pred_token)
                # Just get the Elmo Embedding of the Last Word of the Last Sentence
                curr_i = self._embed_doc([generated_summ_tokens[-1]])[-1].reshape(1, 1, -1)

                # Start a New Line
                if curr_pred_token == '.':
                    generated_summ_tokens.append([])
            else:
                # Init input for prediction
                curr_i = self._embed_doc([generated_summ_tokens[-1]])[-1].reshape(1, 1, -1)

            p_vocab = torch.zeros(size=(1, len(ex_vocab)))
            p_attn = torch.zeros(size=(1, len(ex_vocab)))

            curr_embed_output, curr_deco_state = self.decoder(curr_i, curr_deco_state)

            # Extract the hidden state vector
            curr_deco_hidd, _ = curr_deco_state

            # Calculate Context Vector
            curr_enco_attn = self._align(s=curr_deco_hidd.squeeze(dim=1),
                                         h=encoder_states,
                                         alignment_model=self.alignment_model)
            curr_enco_attn = self.sm_dim0(curr_enco_attn)
            curr_enco_ctxt = torch.matmul(curr_enco_attn, encoder_states)

            # Concatenate Context & Decoder Hidden State
            state_ctxt_concat = torch.cat((curr_deco_hidd.squeeze(), curr_enco_ctxt))

            # Project to Vocabulary
            vocab_prjtn = self.Vocab_Project_2(self.Vocab_Project_1(state_ctxt_concat))
            p_vocab[:, :self.VOCAB_SIZE] = vocab_prjtn
            for src_word, src_attn in zip(orig_tokens_flat, curr_enco_attn):
                p_attn[:, ex_vocab_2_ix[src_word]] += src_attn

            p_gen = self.sigmoid(
                self.Wh_pgen(curr_enco_ctxt) + self.Ws_pgen(curr_deco_hidd.squeeze()) + self.Wx_pgen(
                    curr_i.squeeze()))

            p_W = p_gen * p_vocab + (1 - p_gen) * p_attn
            curr_pred_token = ex_ix_2_vocab[p_W.argmax(dim=1).item()]

            vocab_prjtns.append(p_W)

        vocab_prjtns = torch.cat(vocab_prjtns, dim=0)
        return (vocab_prjtns, ex_vocab_2_ix, ex_ix_2_vocab)

    def forwardx(self, orig_text_tokens: List[List[str]], **kwargs) -> Union:
        # Embed the Orig with Elmo
        orig_embedded_elmo = self._embed_doc(orig_text_tokens)

        # Encode with BiLSTM
        orig_embedded_elmo.unsqueeze_(dim=1)
        encoder_states = self._run_through_bilstm(orig_embedded_elmo, self.encoder)

        # summ_text implies training
        summ_text_tokens = kwargs.get('summ_text_tokens', None)
        summary_length = kwargs.get('summ_text_length', None)

        if summ_text_tokens:
            # -> Training Loop
            print("Training")
            prjtns, v2i, i2v = self._decoder_train(encoder_states=encoder_states,
                                                   src_tokens=orig_text_tokens,
                                                   tgt_tokens=summ_text_tokens)
        else:
            # -> Inference Loop
            print("Testing")
            target_length = 30
            if summary_length:
                target_length = summary_length
            prjtns, v2i, i2v = self._decoder_test(encoder_states=encoder_states,
                                                  src_tokens=orig_text_tokens,
                                                  len_of_summary=target_length)
        return (prjtns, v2i, i2v)
