import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable


class BIMPM(nn.Module):
    def __init__(self, args, data):
        super(BIMPM, self).__init__()

        self.args = args
        #self.d = self.args.word_dim + self.args.char_hidden_size
        self.d = self.args.word_dim
        self.l = self.args.num_perspective

        # ----- Word Representation Layer -----

        #self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim)

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
                    Variable(torch.rand(self.args.hidden_size, self.l)))

        # ----- Aggregation Layer -----

        self.aggregation_LSTM = nn.LSTM(
            input_size=self.l * 8,
            hidden_size=self.args.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- Prediction Layer -----

        self.pred1 = nn.Linear(4 * self.args.hidden_size, self.args.hidden_size)
        self.pred2 = nn.Linear(self.args.hidden_size, self.args.class_size)

    def reset_parameters(self):
        nn.init.uniform(self.char_embedding.weight.data, -0.1, 0.1)

    def forward(self, p, h):
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
            w = w.view(1, 1, self.hidden_size, self.l)

            # (batch, l, seq_len, hidden_size)
            v1, v2 = (v1 * w).permute(0, 3, 1, 2), v2 * w.permute(0, 3, 1, 2)

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
        att_h_forward = con_h_forward.unsqeeze(1) * att_forward.unsqeeze(3)
        att_h_backward = con_h_backward.unsqeeze(1) * att_backward.unsqeeze(3)
        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_p_forward = con_p_forward.unsqeeze(2) * att_forward.unsqeeze(3)
        att_p_backward = con_p_backward.unsqeeze(2) * att_backward.unsqeeze(3)

        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
        att_mean_h_forward = att_h_forward.sum(dim=2) / att_forward.sum(dim=2, keepdim=True)
        att_mean_h_backward = att_h_backward.sum(dim=2) / att_backward.sum(dim=2, keepdim=True)

        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
        att_mean_p_forward = att_p_forward.sum(dim=1) / att_forward.sum(dim=1, keepdim=True).permute(0, 2, 1)
        att_mean_p_backward = att_p_backward.sum(dim=1) / att_backward.sum(dim=1, keepdim=True).permute(0, 2, 1)

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

        # ----- Aggregation Layer -----

        # (batch, seq_len, l * 8) -> (batch, hidden_size * 2)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)

        # (batch, hidden_size * 4)
        x = torch.cat([agg_p_last, agg_h_last], dim=1)

        # ----- Prediction Layer -----

        x = F.relu(self.pred1(x))
        x = self.pred2(x)

        return x


