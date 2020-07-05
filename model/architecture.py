from typing import List, Union
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

        tgt_elmo = self._embed_doc(tgt_tokens, prepend_START=True)

        flat_src_tokens = [i for j in src_tokens for i in j]

        assert len(flat_src_tokens) == encoder_states.shape[0]

        new_words = sorted(list(set([w for w in flat_src_tokens if w not in self.vocab])))

        extended_vocab = self.vocab + new_words
        extended_vocab_2_ix = {**self.vocab_2_ix, **{w: ix for w, ix in zip(new_words, range(
            len(self.vocab), len(extended_vocab)))}}
        extended_ix_2_vocab = {v: k for k, v in extended_vocab_2_ix.items()}

        assert len(extended_vocab) == len(extended_vocab_2_ix) == len(extended_ix_2_vocab)

        # To calculate loss
        collect_xtnd_vocab_prjtns = []

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

        # for curr_elmo in tgt_elmo:
        collected_summary_tokens = [['<START>']]

        curr_elmo = self._embed_doc(doc_tokens=collected_summary_tokens)

        collect_xtnd_vocab_prjtns = []

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

    def forward(self, orig_text_tokens: List[List[str]], **kwargs) -> Union:
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


init_vocab = []
with open('init_vocab_str.txt') as f:
    for line in f.readlines():
        init_vocab.append(line.strip())

assert init_vocab[constant.UNK_TOK_IX] == "<UNK>", "<UNK> Token not found at 0 position "

model = PointerGenerator(vocab=init_vocab,
                         alignment_model="additive",
                         elmo_embed_dim=constant.ELMO_EMBED_DIM,
                         elmo_weights_file=constant.ELMO_WEIGHTS_FILE,
                         elmo_options_file=constant.ELMO_OPTIONS_FILE)

input_texts = [
    "Alexander Haig, who managed the Nixon administration during the Watergate crisis and served a controversial stint as secretary of state under President Reagan, died on Saturday. He was 85. Haig died at Johns Hopkins Hospital in Baltimore, Maryland, after he was admitted there on January 28, spokesman Gary Stephenson said. He served his country well. For that he should be remembered, said William Bennett, who was secretary of education during the Reagan administration. He carried himself well. He carried himself with dignity and honor. The White House issued a statement mourning Haig, saying he exemplified our finest warrior-diplomat tradition of those who dedicate their lives to public service. A top official in the administrations of three presidents Richard Nixon, Gerald Ford and Ronald Reagan Haig served as Nixon s chief of staff during the Watergate political crisis, a scandal that dogged the administration in the 1970s. There was a time during the Watergate crisis when President Nixon was nearly incapacitated, said political analyst and CNN contributor David Gergen, who worked with Haig during the Nixon and Reagan administrations. He had a hard time focusing, so obviously obsessed with the scandal and the gathering storms around him. I watched Al Haig keep the government moving. I thought it was a great act of statesmanship and service to the country. Haig became secretary of state during the Reagan administration and drew controversy for his much-criticized remark on television after the president was shot and wounded by John Hinckley in March 1981. As of now, I am in control here in the White House, Haig said as Vice President George H.W. Bush was headed to Washington from Texas. Haig said he was nt bypassing the rules ; he was just trying to manage the crisis until the vice president arrived. However, he was highly criticized for his behavior, and many observers believe it doomed his political ambitions. Born December 2, 1924, in Bala Cynwyd, Pennsylvania, a suburb of Philadelphia, Alexander Meigs Haig Jr. was raised by his mother after he lost his father at age 10. He attended the University of Notre Dame for two years before transferring to the U.S. Military Academy in 1944. After his graduation in 1947, he served in Japan and later served on Gen. Douglas MacArthur s staff in Japan during the Korean War. He also served in Vietnam, where he earned the distinguished service cross for heroism in combat. He also won the Purple Heart and Silver Star twice. Haig served as supreme allied commander of NATO forces in Europe for five years. There was an assassination attempt on him in Brussels in 1979 as he was being driven to NATO headquarters. A public official known for his loyalty, Haig had hawkish foreign policy views, and Gergen said he could be tough and combustible. He was first and foremost a soldier, Gergen said. Haig was assistant to National Security Adviser Henry Kissinger in the Nixon White House and was involved in the Paris peace agreements that brought an end to the U.S. involvement in the Vietnam War. He was long rumored to be Deep Throat, the Washington Post s inside source on the Watergate break-in and cover-up that eventually destroyed Nixon s presidency. W. Mark Felt, then a high-ranking FBI official, declared in 2005 that he was the source. Great tensions in the Reagan administration simmered over his stances, and Gergen said, There was a sense in the White House that he was grabbing too much power. He wanted to be the, quote, vicar of foreign policy, and there was a lot of pushback from the White House on that. He felt that he had been guaranteed by Ronald Reagan a role as a strong secretary of state and the reins of power would be in his hands. He resented the White House staff trying to manage him, Gergen said. My own sense is that he has been underappreciated, he said. TIME : Read why Haig left the Reagan White House As secretary of state, Haig tried shuttle diplomacy to head off war between Britain and Argentina over the Falkland Islands in 1982, but he failed. He opposed Reagan s handling of Iran and disagreed with the president s plan on aid to the contra rebels in Nicaragua. He eventually left the Reagan administration after 18 months and made a run for president in 1988, pulling out before the New Hampshire primary. He backed Bob Dole instead of George H.W. Bush when he dropped out. Former U.S. Ambassador to the U.N. John Bolton announced Haig s death to the Conservative Political Action Conference in Washington on Saturday and called him a patriot ."]
output_texts = [
    "Haig worked under Presidents Nixon, Ford, Reagan. He was highly decorated soldier who served during Korean and Vietnam wars. As secretary of state, Haig wrongly declared I am in control here after Reagan was shot. He unsuccessfully sought the 1988 Republican presidential nomination"]

def train(orig_text, summ_text, curr_model: PointerGenerator):
    orig_tokens = helper.tokenize_en(orig_text, lowercase=True)
    summ_tokens = helper.tokenize_en(summ_text, lowercase=True)
    prjtns, v2i, i2v = curr_model(orig_text_tokens=orig_tokens,
                                  summ_text_tokens=summ_tokens)
    flat_summ_tokens = [i for j in summ_tokens for i in j]
    gold_ixs = torch.LongTensor([v2i.get(w, constant.UNK_TOK_IX) for w in flat_summ_tokens])
    loss = loss_fn(input=prjtns, target=gold_ixs)
    print("LOSS -> {0}".format(loss.item()))
    return loss


def test(orig_text, summ_text_length, curr_model: PointerGenerator):
    # Tokenize by Sents
    orig_tokens = helper.tokenize_en(orig_text, lowercase=True)
    prjtns, v2i, i2v = curr_model(orig_text_tokens=orig_tokens, summ_text_length=summ_text_length)
    prjtns_argmax = prjtns.argmax(dim=1)
    pred_summary = ' '.join([i2v[i.item()] for i in prjtns_argmax])
    print("Predicted Summary : \n{0}".format(pred_summary))
    return pred_summary


# x = train(orig_text=input_texts[0], summ_text=output_texts[0], curr_model=model)

x = test(orig_text=input_texts[0], summ_text_length=9, curr_model=model)
