import torch
import torch.nn as nn
import torch.nn.functional as F


def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # <pad>
    return pad_attn_mask


def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1)  # upper triangular part of a matrix(2-D)
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_head ** 0.5)

    def forward(self, Q, K, V, attn_mask):
        # scores=(bs, n_head, n_enc_seq, n_enc_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        # attn_prob=(bs, n_head, n_enc_seq, n_enc_seq)=scores
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        # context=(bs, n_head, n_enc_seq, d_head)=(bs, n_head, n_enc_seq, n_enc_seq) * (bs, n_head, ne_enc_seq, d_head)
        context = torch.matmul(attn_prob, V)
        return context, attn_prob


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_K = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_V = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_hidn)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        # q_s = (bs, n_head, n_enc_seq, d_head)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        # k_s = (bs, n_head, n_enc_seq, d_head)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        # v_s = (bs, n_head, n_v_seq, d_head)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)

        # attn_mask = (bs, n_head, n_enc_seq, n_enc_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

        # context = (bs, n_head, n_enc_seq, d_head), attn_prob = (bs, n_head, n_enc_seq, n_enc_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        # context = (bs, n_enc_seq, n_head, d_head)
        context = context.transpose(1, 2).contiguous()
        # context = (bs, n_enc_seq, n_head * d_head)
        context = context.view(batch_size, -1, self.config.n_head * self.config.d_head)
        # output = (bs, n_head, n_enc_seq, d_hidn)
        output = self.linear(context)  # n_head * d_head -> d_hidn
        output = self.dropout(output)
        # output = (bs, n_enc_seq, d_hidn), attn_prob = (bs, n_head, n_enc_seq, n_enc_seq)
        return output, attn_prob


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidn, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidn, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # inputs=(bs, n_enc_seq, d_hidn)
        # output=(bs, d_ff, n_enc_seq), inputs.transpose(1, 2)=(bs, d_hidn, n_enc_seq)
        output = self.conv1(inputs.transpose(1, 2))
        output = self.active(output)
        # output=(bs, n_enc_seq, d_hidn)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)

    def forward(self, inputs, attn_mask):
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        # (bs, n_enc_seq, d_hidn)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        return ffn_outputs, attn_prob


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # V=n_enc_vocab, H=d_hidn => BERT(V*E)
        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidn)
        self.pos_emb = nn.Embedding(self.config.n_enc_seq + 1, self.config.d_hidn)
        self.seg_emb = nn.Embedding(self.config.n_seg_type, self.config.d_hidn)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, inputs, segments):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(self.config.i_pad)
        # positions = (bs, n_enc_seq)
        positions.masked_fill_(pos_mask, 0)

        # outputs = (bs, n_enc_seq, d_hidn)
        outputs = self.enc_emb(inputs) + self.pos_emb(positions) + self.seg_emb(segments)

        # attn_mask = (bs, n_enc_seq, n_enc_seq)
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)

        attn_probs = []
        for layer in self.layers:
            # outputs = (bs, n_enc_seq, d_hidn), attn_prob = (bs, n_head, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        # outputs = (bs, n_enc_seq, d_hidn), attn_probs = [(bs, n_head, n_enc_seq, n_enc_seq)]
        return outputs, attn_probs


class PooledOutput(nn.Module):
    """ huggingface's pooled output """

    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.d_hidn, config.d_hidn)
        self.activation = torch.tanh

    def forward(self, inputs):
        outputs_cls = self.linear(inputs)
        outputs_cls = self.activation(outputs_cls)
        return outputs_cls


class BERT(nn.Module):
    def __init__(self, config, use_pooled=True):
        super().__init__()
        self.encoder = Encoder(config)
        self.use_pooled = use_pooled
        if use_pooled:
            self.pooled = PooledOutput(config)

    def forward(self, inputs, segments):
        outputs, self_attn_probs = self.encoder(inputs, segments)
        if self.use_pooled:
            outputs_cls = self.pooled(outputs[:, 0].contiguous())
        else:
            outputs_cls = outputs[:, 0].contiguous()
        # outputs = (bs, n_enc_seq, d_hidn), outputs_cls = (bs, d_hidn), attn_probs = [(bs, n_head, n_enc_seq, n_enc_seq)]
        return outputs, outputs_cls, self_attn_probs

    def save(self, epoch, loss, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "state_dict": self.state_dict()
        }, path)

    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"]


class BERTPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert = BERT(self.config)
        # classfier
        self.projection_cls = nn.Linear(self.config.d_hidn, 2)  # Next Sentence Prediction
        # lm
        self.projection_lm = nn.Linear(self.config.d_hidn, self.config.n_enc_vocab, bias=False)
        with torch.no_grad():
            self.projection_lm.weight = self.bert.encoder.enc_emb.weight

    def forward(self, inputs, segments):
        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        outputs, outputs_cls, attn_probs = self.bert(inputs, segments)
        # (bs, d_hidn)
        logits_cls = self.projection_cls(outputs_cls)
        # (bs, n_enc_seq, n_enc_vocab)
        logits_lm = self.projection_lm(outputs)
        # (bs, n_enc_vocab), (bs, n_enc_seq, n_enc_vocab), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return logits_cls, logits_lm, attn_probs


class BertTrainMovie(nn.Module):
    """ naver movie classfication """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert = BERT(self.config)
        # classfier
        self.projection_cls = nn.Linear(self.config.d_hidn, self.config.n_output, bias=False)

    def forward(self, inputs, segments):
        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        outputs, outputs_cls, attn_probs = self.bert(inputs, segments)
        # (bs, n_output)
        logits_cls = self.projection_cls(outputs_cls)
        # (bs, n_output), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return logits_cls, attn_probs

    def save(self, epoch, loss, score, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "score": score,
            "state_dict": self.state_dict()
        }, path)

    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"], save["score"]
