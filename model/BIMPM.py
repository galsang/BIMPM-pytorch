import torch
import torch.nn as nn
import torch.nn.functional as F


class BIMPM(nn.Module):
    def __init__(self, args, data):
        super(BIMPM, self).__init__()

        self.args = args
        self.d = self.args.word_dim + int(self.args.use_char_emb) * self.args.char_hidden_size
        self.l = self.args.num_perspective

        # ----- Word Representation Layer -----
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=0)

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
            setattr(self, f'mp_w{i}',
                    nn.Parameter(torch.rand(self.l, self.args.hidden_size)))

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

        # <unk> vectors is randomly initialized
        nn.init.uniform(self.word_emb.weight.data[0], -0.1, 0.1)

        nn.init.kaiming_normal(self.char_LSTM.weight_ih_l0)
        nn.init.constant(self.char_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.char_LSTM.weight_hh_l0)
        nn.init.constant(self.char_LSTM.bias_hh_l0, val=0)

        # ----- Context Representation Layer -----
        nn.init.kaiming_normal(self.context_LSTM.weight_ih_l0)
        nn.init.constant(self.context_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.context_LSTM.weight_hh_l0)
        nn.init.constant(self.context_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal(self.context_LSTM.weight_ih_l0_reverse)
        nn.init.constant(self.context_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal(self.context_LSTM.weight_hh_l0_reverse)
        nn.init.constant(self.context_LSTM.bias_hh_l0_reverse, val=0)

        # ----- Matching Layer -----
        for i in range(1, 9):
            w = getattr(self, f'mp_w{i}')
            nn.init.kaiming_normal(w)

        # ----- Aggregation Layer -----
        nn.init.kaiming_normal(self.aggregation_LSTM.weight_ih_l0)
        nn.init.constant(self.aggregation_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.aggregation_LSTM.weight_hh_l0)
        nn.init.constant(self.aggregation_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal(self.aggregation_LSTM.weight_ih_l0_reverse)
        nn.init.constant(self.aggregation_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal(self.aggregation_LSTM.weight_hh_l0_reverse)
        nn.init.constant(self.aggregation_LSTM.bias_hh_l0_reverse, val=0)

        # ----- Prediction Layer ----
        nn.init.uniform(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant(self.pred_fc1.bias, val=0)

        nn.init.uniform(self.pred_fc2.weight, -0.005, 0.005)
        nn.init.constant(self.pred_fc2.bias, val=0)

    def dropout(self, v):
        return F.dropout(v, p=self.args.dropout, training=self.args.is_training)

    def forward(self, **kwargs):
        # ----- Matching Layer -----
        def mp_matching_func(v1, v2, w):
            """
            :param v1: (batch, seq_len, hidden_size)
            :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
            :param w: (l, hidden_size)
            :return: (batch, l)
            """
            seq_len = v1.size(1)

            # Trick for large memory requirement
            """
            if len(v2.size()) == 2:
                v2 = torch.stack([v2] * seq_len, dim=1)

            m = []
            for i in range(self.l):
                # v1: (batch, seq_len, hidden_size)
                # v2: (batch, seq_len, hidden_size)
                # w: (1, 1, hidden_size)
                # -> (batch, seq_len)
                m.append(F.cosine_similarity(w[i].view(1, 1, -1) * v1, w[i].view(1, 1, -1) * v2, dim=2))

            # list of (batch, seq_len) -> (batch, seq_len, l)
            m = torch.stack(m, dim=2)
            """

            # (1, 1, hidden_size, l)
            w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
            # (batch, seq_len, hidden_size, l)
            v1 = w * torch.stack([v1] * self.l, dim=3)
            if len(v2.size()) == 3:
                v2 = w * torch.stack([v2] * self.l, dim=3)
            else:
                v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * self.l, dim=3)

            m = F.cosine_similarity(v1, v2, dim=2)

            return m

        def mp_matching_func_pairwise(v1, v2, w):
            """
            :param v1: (batch, seq_len1, hidden_size)
            :param v2: (batch, seq_len2, hidden_size)
            :param w: (l, hidden_size)
            :return: (batch, l, seq_len1, seq_len2)
            """

            # Trick for large memory requirement
            """
            m = []
            for i in range(self.l):
                # (1, 1, hidden_size)
                w_i = w[i].view(1, 1, -1)
                # (batch, seq_len1, hidden_size), (batch, seq_len2, hidden_size)
                v1, v2 = w_i * v1, w_i * v2
                # (batch, seq_len, hidden_size->1)
                v1_norm = v1.norm(p=2, dim=2, keepdim=True)
                v2_norm = v2.norm(p=2, dim=2, keepdim=True)

                # (batch, seq_len1, seq_len2)
                n = torch.matmul(v1, v2.permute(0, 2, 1))
                d = v1_norm * v2_norm.permute(0, 2, 1)

                m.append(div_with_small_value(n, d))

            # list of (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, l)
            m = torch.stack(m, dim=3)
            """

            # (1, l, 1, hidden_size)
            w = w.unsqueeze(0).unsqueeze(2)
            # (batch, l, seq_len, hidden_size)
            v1, v2 = w * torch.stack([v1] * self.l, dim=1), w * torch.stack([v2] * self.l, dim=1)
            # (batch, l, seq_len, hidden_size->1)
            v1_norm = v1.norm(p=2, dim=3, keepdim=True)
            v2_norm = v2.norm(p=2, dim=3, keepdim=True)

            # (batch, l, seq_len1, seq_len2)
            n = torch.matmul(v1, v2.transpose(2, 3))
            d = v1_norm * v2_norm.transpose(2, 3)

            # (batch, seq_len1, seq_len2, l)
            m = div_with_small_value(n, d).permute(0, 2, 3, 1)

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
            d = v1_norm * v2_norm

            return div_with_small_value(a, d)

        def div_with_small_value(n, d, eps=1e-8):
            # too small values are replaced by 1e-8 to prevent it from exploding.
            d = d * (d > eps).float() + eps * (d <= eps).float()
            return n / d

        # ----- Word Representation Layer -----
        # (batch, seq_len) -> (batch, seq_len, word_dim)

        p = self.word_emb(kwargs['p'])
        h = self.word_emb(kwargs['h'])

        if self.args.use_char_emb:
            # (batch, seq_len, max_word_len) -> (batch * seq_len, max_word_len)
            seq_len_p = kwargs['char_p'].size(1)
            seq_len_h = kwargs['char_h'].size(1)

            char_p = kwargs['char_p'].view(-1, self.args.max_word_len)
            char_h = kwargs['char_h'].view(-1, self.args.max_word_len)

            # (batch * seq_len, max_word_len, char_dim)-> (1, batch * seq_len, char_hidden_size)
            _, (char_p, _) = self.char_LSTM(self.char_emb(char_p))
            _, (char_h, _) = self.char_LSTM(self.char_emb(char_h))

            # (batch, seq_len, char_hidden_size)
            char_p = char_p.view(-1, seq_len_p, self.args.char_hidden_size)
            char_h = char_h.view(-1, seq_len_h, self.args.char_hidden_size)

            # (batch, seq_len, word_dim + char_hidden_size)
            p = torch.cat([p, char_p], dim=-1)
            h = torch.cat([h, char_h], dim=-1)

        p = self.dropout(p)
        h = self.dropout(h)

        # ----- Context Representation Layer -----
        # (batch, seq_len, hidden_size * 2)
        con_p, _ = self.context_LSTM(p)
        con_h, _ = self.context_LSTM(h)

        con_p = self.dropout(con_p)
        con_h = self.dropout(con_h)

        # (batch, seq_len, hidden_size)
        con_p_fw, con_p_bw = torch.split(con_p, self.args.hidden_size, dim=-1)
        con_h_fw, con_h_bw = torch.split(con_h, self.args.hidden_size, dim=-1)

        # 1. Full-Matching

        # (batch, seq_len, hidden_size), (batch, hidden_size)
        # -> (batch, seq_len, l)
        mv_p_full_fw = mp_matching_func(con_p_fw, con_h_fw[:, -1, :], self.mp_w1)
        mv_p_full_bw = mp_matching_func(con_p_bw, con_h_bw[:, 0, :], self.mp_w2)
        mv_h_full_fw = mp_matching_func(con_h_fw, con_p_fw[:, -1, :], self.mp_w1)
        mv_h_full_bw = mp_matching_func(con_h_bw, con_p_bw[:, 0, :], self.mp_w2)

        # 2. Maxpooling-Matching

        # (batch, seq_len1, seq_len2, l)
        mv_max_fw = mp_matching_func_pairwise(con_p_fw, con_h_fw, self.mp_w3)
        mv_max_bw = mp_matching_func_pairwise(con_p_bw, con_h_bw, self.mp_w4)

        # (batch, seq_len, l)
        mv_p_max_fw, _ = mv_max_fw.max(dim=2)
        mv_p_max_bw, _ = mv_max_bw.max(dim=2)
        mv_h_max_fw, _ = mv_max_fw.max(dim=1)
        mv_h_max_bw, _ = mv_max_bw.max(dim=1)

        # 3. Attentive-Matching

        # (batch, seq_len1, seq_len2)
        att_fw = attention(con_p_fw, con_h_fw)
        att_bw = attention(con_p_bw, con_h_bw)

        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
        att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)

        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
        att_mean_h_fw = div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        att_mean_h_bw = div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))

        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
        att_mean_p_fw = div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        att_mean_p_bw = div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))

        # (batch, seq_len, l)
        mv_p_att_mean_fw = mp_matching_func(con_p_fw, att_mean_h_fw, self.mp_w5)
        mv_p_att_mean_bw = mp_matching_func(con_p_bw, att_mean_h_bw, self.mp_w6)
        mv_h_att_mean_fw = mp_matching_func(con_h_fw, att_mean_p_fw, self.mp_w5)
        mv_h_att_mean_bw = mp_matching_func(con_h_bw, att_mean_p_bw, self.mp_w6)

        # 4. Max-Attentive-Matching

        # (batch, seq_len1, hidden_size)
        att_max_h_fw, _ = att_h_fw.max(dim=2)
        att_max_h_bw, _ = att_h_bw.max(dim=2)
        # (batch, seq_len2, hidden_size)
        att_max_p_fw, _ = att_p_fw.max(dim=1)
        att_max_p_bw, _ = att_p_bw.max(dim=1)

        # (batch, seq_len, l)
        mv_p_att_max_fw = mp_matching_func(con_p_fw, att_max_h_fw, self.mp_w7)
        mv_p_att_max_bw = mp_matching_func(con_p_bw, att_max_h_bw, self.mp_w8)
        mv_h_att_max_fw = mp_matching_func(con_h_fw, att_max_p_fw, self.mp_w7)
        mv_h_att_max_bw = mp_matching_func(con_h_bw, att_max_p_bw, self.mp_w8)

        # (batch, seq_len, l * 8)
        mv_p = torch.cat(
            [mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,
             mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw], dim=2)
        mv_h = torch.cat(
            [mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,
             mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw], dim=2)

        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        # ----- Aggregation Layer -----
        # (batch, seq_len, l * 8) -> (2, batch, hidden_size)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)

        # 2 * (2, batch, hidden_size) -> 2 * (batch, hidden_size * 2) -> (batch, hidden_size * 4)
        x = torch.cat(
            [agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2),
             agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2)], dim=1)
        x = self.dropout(x)

        # ----- Prediction Layer -----
        x = F.tanh(self.pred_fc1(x))
        x = self.dropout(x)
        x = self.pred_fc2(x)

        return x
