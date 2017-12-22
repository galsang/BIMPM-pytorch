import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable


class BIMPM(nn.Module):
    """
    TODO: character embedding implementation
    """

    def __init__(self, args, data):
        super(BIMPM, self).__init__()

        self.args = args
        self.d = self.args.word_dim + int(self.args.use_char_emb) * self.args.char_hidden_size
        self.l = self.args.num_perspective

        # ----- Word Representation Layer -----

        self.char_emb = nn.Embedding(args.char_vocab_size + 1, args.char_dim, padding_idx=0)

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
            input_size=self.d,
            hidden_size=self.args.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- Matching Layer -----

        for i in range(1, 9):
            setattr(self, f'mp_weight{i}',
                    nn.Parameter(torch.rand(self.args.hidden_size, self.l)))

        # ----- Aggregation Layer -----

        self.aggregation_LSTM = nn.LSTM(
            input_size=self.l * 8,
            hidden_size=self.args.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- Prediction Layer -----

        self.pred_fc1 = nn.Linear(self.args.hidden_size * 4, self.args.hidden_size * 2)
        self.pred_fc2 = nn.Linear(self.args.hidden_size * 2, self.args.class_size)

        self.reset_parameters()

    def reset_parameters(self):
        # ----- Word Representation Layer -----
        nn.init.uniform(self.char_emb.weight, -0.005, 0.005)
        # zero vectors for padding
        self.char_emb.weight.data[0].fill_(0)

        nn.init.kaiming_normal(self.char_LSTM.weight_ih_l0)
        nn.init.constant(self.char_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.char_LSTM.weight_hh_l0)
        nn.init.constant(self.char_LSTM.bias_hh_l0, val=0)

        # ----- Context Representation Layer -----

        nn.init.kaiming_normal(self.context_LSTM.weight_ih_l0)
        nn.init.constant(self.context_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.context_LSTM.weight_hh_l0)
        nn.init.constant(self.context_LSTM.bias_hh_l0, val=0)

        # ----- Matching Layer -----

        for i in range(1, 9):
            w = getattr(self, f'mp_weight{i}')
            nn.init.kaiming_normal(w)

        # ----- Aggregation Layer -----

        nn.init.kaiming_normal(self.aggregation_LSTM.weight_ih_l0)
        nn.init.constant(self.aggregation_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.aggregation_LSTM.weight_hh_l0)
        nn.init.constant(self.aggregation_LSTM.bias_hh_l0, val=0)

        # ----- Prediction Layer ----

        nn.init.uniform(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant(self.pred_fc1.bias, val=0)

        nn.init.uniform(self.pred_fc2.weight, -0.005, 0.005)
        nn.init.constant(self.pred_fc2.bias, val=0)

    def dropout(self, v):
        return F.dropout(v, p=self.args.dropout, training=self.training)

    def forward(self, p, h):
        # ----- Word Representation Layer -----

        # (batch, seq_len) -> (batch, seq_len, word_dim)
        p = self.word_emb(p)
        h = self.word_emb(h)

        p = self.dropout(p)
        h = self.dropout(h)

        # ----- Context Representation Layer -----

        # (batch, seq_len, hidden_size * 2)
        con_p, _ = self.context_LSTM(p)
        con_h, _ = self.context_LSTM(h)

        con_p = self.dropout(con_p)
        con_h = self.dropout(con_h)

        # (batch, seq_len, hidden_size)
        con_p_forward = con_p[:, :, :self.args.hidden_size]
        con_p_backward = con_p[:, :, self.args.hidden_size:]
        con_h_forward = con_h[:, :, :self.args.hidden_size]
        con_h_backward = con_h[:, :, self.args.hidden_size:]

        # ----- Matching Layer -----

        def mp_matching_func(v1, v2, w):
            """
            :param v1: (batch, seq_len1, hidden_size)
            :param v2: (batch, seq_len2, hidden_size) or (batch, hidden_size)
            :param w: (hidden_size, l)
            :return: (batch, l, seq_len1, seq_len2)
            """

            # (batch, seq_len1, hidden_size, l)
            v1 = torch.stack([v1] * self.l, dim=3)
            # (batch, seq_len2, hidden_size, l)
            v2 = torch.stack([v2] * self.l, dim=3)
            # (1, 1, hidden_size, l)
            w = w.view(1, 1, self.args.hidden_size, self.l)

            # (batch, l, seq_len, hidden_size)
            v1, v2 = (v1 * w).permute(0, 3, 1, 2), (v2 * w).permute(0, 3, 1, 2)

            # (batch, l, seq_len1, 1)
            v1_norm = v1.norm(p=2, dim=3, keepdim=True)
            # (batch, l, 1, seq_len2)
            v2_norm = v2.norm(p=2, dim=3, keepdim=True).permute(0, 1, 3, 2)

            # (batch, l, seq_len1, seq_len2)
            m = torch.matmul(v1, v2.permute(0, 1, 3, 2))
            m /= v1_norm * v2_norm

            return m

        def attention(v1, v2):
            """
            :param v1: (batch, seq_len1, hidden_size)
            :param v2: (batch, seq_len2, hidden_size)
            :return: (batch, seq_len1, seq_len2)
            """

            # (batch, seq_len1, 1)
            v1_norm = v1.norm(p=2, dim=2, keepdim=True)
            # (batch, 1, seq_len2)
            v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

            # (batch, seq_len1, seq_len2)
            a = torch.bmm(v1, v2.permute(0, 2, 1))
            a /= v1_norm * v2_norm

            return a

        # 1. Full-Matching

        # (batch, l, seq_len1, seq_len2)
        mv_full_forward = mp_matching_func(con_p_forward, con_h_forward, self.mp_weight1)
        mv_full_backward = mp_matching_func(con_p_backward, con_h_backward, self.mp_weight2)

        # (batch, l, seq_len)
        mv_p_full_forward = mv_full_forward[:, :, :, -1]
        mv_p_full_backward = mv_full_backward[:, :, :, 0]
        mv_h_full_forward = mv_full_forward[:, :, -1, :]
        mv_h_full_backward = mv_full_backward[:, :, 0, :]

        # 2. Maxpooling-Matching

        # (batch, l, seq_len1, seq_len2)
        mv_max_forward = mp_matching_func(con_p_forward, con_h_forward, self.mp_weight3)
        mv_max_backward = mp_matching_func(con_p_backward, con_h_backward, self.mp_weight4)

        # (batch, l, seq_len)
        mv_p_max_forward, _ = mv_max_forward.max(dim=3)
        mv_p_max_backward, _ = mv_max_backward.max(dim=3)
        mv_h_max_forward, _ = mv_max_forward.max(dim=2)
        mv_h_max_backward, _ = mv_max_backward.max(dim=2)

        # 3. Attentive-Matching

        # (batch, seq_len1, seq_len2)
        att_forward = attention(con_p_forward, con_h_forward)
        att_backward = attention(con_p_backward, con_h_backward)

        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_h_forward = con_h_forward.unsqueeze(1) * att_forward.unsqueeze(3)
        att_h_backward = con_h_backward.unsqueeze(1) * att_backward.unsqueeze(3)
        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_p_forward = con_p_forward.unsqueeze(2) * att_forward.unsqueeze(3)
        att_p_backward = con_p_backward.unsqueeze(2) * att_backward.unsqueeze(3)

        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
        att_mean_h_forward = att_h_forward.sum(dim=2)
        att_mean_h_forward /= att_forward.sum(dim=2, keepdim=True) + 1e-10
        att_mean_h_backward = att_h_backward.sum(dim=2)
        att_mean_h_backward /= att_backward.sum(dim=2, keepdim=True) + 1e-10

        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
        att_mean_p_forward = att_p_forward.sum(dim=1)
        att_mean_p_forward /= att_forward.sum(dim=1, keepdim=True).permute(0, 2, 1) + 1e-10
        att_mean_p_backward = att_p_backward.sum(dim=1)
        att_mean_p_backward /= att_backward.sum(dim=1, keepdim=True).permute(0, 2, 1) + 1e-10

        # (batch, l, seq_len)
        mv_p_att_mean_forward = mp_matching_func(con_p_forward, att_mean_h_forward, self.mp_weight5)[:, :, :, 0]
        mv_p_att_mean_backward = mp_matching_func(con_p_backward, att_mean_h_backward, self.mp_weight6)[:, :, :, 0]
        mv_h_att_mean_forward = mp_matching_func(con_h_forward, att_mean_p_forward, self.mp_weight5)[:, :, :, 0]
        mv_h_att_mean_backward = mp_matching_func(con_h_backward, att_mean_p_backward, self.mp_weight6)[:, :, :, 0]

        # 4. Max-Attentive-Matching

        # (batch, seq_len1, hidden_size)
        att_max_h_forward, _ = att_h_forward.max(dim=2)
        att_max_h_backward, _ = att_h_backward.max(dim=2)
        # (batch, seq_len2, hidden_size)
        att_max_p_forward, _ = att_p_forward.max(dim=1)
        att_max_p_backward, _ = att_p_backward.max(dim=1)

        # (batch, l, seq_len)
        mv_p_att_max_forward = mp_matching_func(con_p_forward, att_max_h_forward, self.mp_weight7)[:, :, :, 0]
        mv_p_att_max_backward = mp_matching_func(con_p_backward, att_max_h_backward, self.mp_weight8)[:, :, :, 0]
        mv_h_att_max_forward = mp_matching_func(con_h_forward, att_max_p_forward, self.mp_weight7)[:, :, :, 0]
        mv_h_att_max_backward = mp_matching_func(con_h_backward, att_max_p_backward, self.mp_weight8)[:, :, :, 0]

        # (batch, seq_len, l * 8)
        mv_p = torch.cat(
            [mv_p_full_forward, mv_p_max_forward, mv_p_att_mean_forward, mv_p_att_max_forward,
             mv_p_full_backward, mv_p_max_backward, mv_p_att_mean_backward, mv_p_att_max_backward],
            dim=1).permute(0, 2, 1)
        mv_h = torch.cat(
            [mv_h_full_forward, mv_h_max_forward, mv_h_att_mean_forward, mv_h_att_max_forward,
             mv_h_full_backward, mv_h_max_backward, mv_h_att_mean_backward, mv_h_att_max_backward],
            dim=1).permute(0, 2, 1)

        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        # ----- Aggregation Layer -----

        # (batch, seq_len, l * 8) -> (2, batch, hidden_size)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)

        # (2, batch, hidden_size) -> (batch, hidden_size * 2)
        agg_p_last = agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2)
        agg_h_last = agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2)
        # (batch, hidden_size * 4)
        x = torch.cat([agg_p_last, agg_h_last], dim=1)
        x = self.dropout(x)

        # ----- Prediction Layer -----

        x = self.dropout(F.tanh(self.pred_fc1(x)))
        x = self.pred_fc2(x)

        return x
