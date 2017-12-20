import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable


class BIMPM(nn.Module):
    def __init__(self, args, data):
        super(BIMPM, self).__init__()

        self.args = args
        self.d = self.args.hidden_size + self.args.char_hidden_size
        self.l = self.args.num_perspective

        # ----- Word Representation Layer -----

        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim)

        self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
        # initialize word embedding with GloVe
        self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
        # no fine-tuning for word vectors
        self.word_emb.weight.requires_grad = False

        self.char_LSTM = nn.LSTM(
            input_size=self.args.char_dim,
            hidden_size=self.args.char_hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True)

        # ----- Context Representation Layer -----

        self.context_LSTM = nn.LSTM(
            input_size=self.args.char_hidden_size + self.args.word_dim,
            hidden_size=self.args.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- Matching Layer -----

        for i in range(1, 7):
            setattr(self, f'mp_weight{i}',
                    Variable(torch.rand(self.args.hidden_size, self.l)))

        # ----- Aggregation Layer -----

        self.aggregation_LSTM = nn.LSTM(
            input_size=self.args.hidden_size * 8,
            hidden_size=self.args.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- Prediction Layer -----

        self.prediction_layer1 = nn.Linear(4 * self.args.hidden_size, self.args.hidden_size)
        self.prediction_layer2 = nn.Linear(self.args.hidden_size, self.args.class_size)

    def reset_parameters(self):
        nn.init.uniform(self.char_embedding.weight.data, -0.1, 0.1)

    def forward(self, p, h):
        def mp_matching_func(v1, v2, w):
            """
            :param v1: (batch, seq_len1, hidden_size)
            :param v2: (batch, seq_len2, hidden_size)
            :param w: (hidden_size, l)
            :return: (batch, l, seq_len1, seq_len2)

            """
            seq_len1, seq_len2 = v1.size()[1], v2.size[1]

            # (batch, seq_len1, hidden_size, l)
            v1 = torch.stack([v1] * self.l, dim=3)
            # (batch, seq_len2, hidden_size, l)
            v2 = torch.stack([v2] * self.l, dim=3)
            w = w.view(1, 1, self.hidden_size, self.l)

            # (batch, l, seq_len, hidden_size)
            v1, v2 = (v1 * w).permute(0, 3, 1, 2), v2 * w.permute(0, 3, 1, 2)

            # (batch, l, seq_len1, 1)
            v1_norm = v1.norm(p=2, dim=3).view(-1, self.l, seq_len1, 1)
            # (batch, l, 1, seq_len2)
            v2_norm = v2.norm(p=2, dim=3).view(-1, self.l, 1, seq_len2)

            # (batch, l, seq_len1, seq_len2)
            m = torch.matmul(v1, v2.permute(0, 1, 3, 2))
            m /= v1_norm
            m /= v2_norm

            return m

        # ----- Word Representation Layer -----

        # (batch, seq_len) -> (batch, seq_len, word_dim)
        p = self.word_emb(p)
        h = self.word_emb(h)

        # ----- Context Representation Layer -----

        # (batch, seq_len, hidden_size * 2)
        con_p, _ = self.context_LSTM(p)
        con_h, _ = self.context_LSTM(h)

        # (batch, seq_len, hidden_size)
        con_p_forward = con_p[:, :, :self.args.hidden_size]
        con_p_backward = con_p[:, :, self.args.hidden_size:]
        con_h_forward = con_h[:, :, :self.args.hidden_size]
        con_h_backward = con_h[:, :, self.args.hidden_size:]

        # ----- Matching Layer -----

        # 1. Full-Matching
        # (batch, l, seq_len1, seq_len2)
        mv_full_forward = mp_matching_func(con_p_forward, con_h_forward, self.mp_weight1)
        # (batch, l, seq_len1, seq_len2)
        mv_full_backward = mp_matching_func(con_p_backward, con_h_backward, self.mp_weight2)

        # (batch, l, seq_len1)
        mv_p_full_forward = mv_full_forward[:, :, :, -1]
        mv_p_full_backward = mv_full_backward[:, :, :, 0]

        # (batch, l, seq_len2)
        mv_h_full_forward = mv_full_forward[:, :, -1, :]
        mv_h_full_backward = mv_full_backward[:, :, 0, :]

        # 2. Maxpooling-Matching

        # (batch, l, seq_len1, seq_len2)
        mv_max_forward = mp_matching_func(con_p_forward, con_h_forward, self.mp_weight3)
        mv_max_backward = mp_matching_func(con_p_backward, con_h_backward, self.mp_weight4)

        # (batch, l, seq_len1)
        mv_p_max_forward = mv_max_forward.max(dim=3)
        mv_p_max_backward = mv_max_backward.max(dim=3)
        # (batch, l, seq_len2)
        mv_h_max_forward = mv_max_forward.max(dim=2)
        mv_h_max_backward = mv_max_backward.max(dim=2)

        # 3. Attentive-Matching

        mv_p_att_forward =

        # 4. Max-Attentive-Matching

        mv_p_max_att_forward =

        # ----- Aggregation Layer -----

        # ----- Prediction Layer -----
