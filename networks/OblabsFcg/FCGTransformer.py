import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """

        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim  # 512 dim
        self.n_heads = n_heads  # 8
        self.single_head_dim = int(self.embed_dim / self.n_heads)  # 512/8 = 64  . each key,query, value will be of 64d

        # key,query and value matrixes    #64 x 64
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim,
                                      bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)
        self.attention_map = None

    def forward(self, key, query, value, mask=None):  # batch_size x sequence_length x embedding_dim    # 32 x 10 x 512

        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder

        Returns:
           output vector from multihead attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)

        # query dimension can change in decoder during inference.
        # so we cant take general seq_length
        seq_length_query = query.size(1)

        # 32x10x512
        key = key.view(batch_size, seq_length, self.n_heads,
                       self.single_head_dim)  # batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)  # (32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  # (32x10x8x64)

        k = self.key_matrix(key)  # (32x10x8x64)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)

        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1, -2)  # (batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  # (32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)

        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
            # product = product.masked_fill(mask == 0, float("-1e20"))
            product = product.masked_fill(mask == 0, float("0.0"))

        # divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim)  # / sqrt(64)

        # applying softmax
        scores = F.softmax(product, dim=-1)
        self.attention_map = scores

        # mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64)

        # concatenated output
        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_length_query,
                                                          self.single_head_dim * self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)

        # Return different value
        output = self.out(concat)  # (32,10,512) -> (32,10,512)

        return output


class SelfAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(SelfAttention, self).__init__()

        self.embed_dim = embed_dim  # 512 dim
        self.n_heads = n_heads  # 8
        self.single_head_dim = int(self.embed_dim / self.n_heads)  # 512/8 = 64  . each key,query, value will be of 64d

        # key,query and value matrixes    #64 x 64
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim,
                                      bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(self, x, mask=None):  # batch_size x sequence_length x embedding_dim    # 32 x 10 x 512
        batch_size = x.size(0)
        seq_length = x.size(1)

        # query dimension can change in decoder during inference.
        # so we cant take general seq_length
        seq_length_query = x.size(1)

        # 32x10x512
        key = x.view(batch_size, seq_length, self.n_heads,
                     self.single_head_dim)  # batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = x.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)  # (32x10x8x64)
        value = x.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  # (32x10x8x64)

        k = self.key_matrix(key)  # (32x10x8x64)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)

        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1, -2)  # (batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  # (32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)

        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        # divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim)  # / sqrt(64)

        # applying softmax
        scores = F.softmax(product, dim=-1)

        # mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64)

        # concatenated output
        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_length_query,
                                                          self.single_head_dim * self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)

        output = self.out(concat)  # (32,10,512) -> (32,10,512)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8, p=0.1):
        super(TransformerBlock, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads

        """
        self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)

    def forward(self, key, query, value):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block

        """

        attention_out = self.attention(key, query, value)  # 32x10x512
        # attention_residual_out = attention_out + value  # 32x10x512
        attention_residual_out = attention_out + query  # 32x10x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out))  # 32x10x512

        feed_fwd_out = self.feed_forward(norm1_out)  # 32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out  # 32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))  # 32x10x512

        return norm2_out


class PatchEmbed(nn.Module):
    def __init__(self, signal_size, patch_size, in_chans=1, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.signal_size = signal_size
        self.patch_size = patch_size
        self.n_patches = (signal_size // patch_size)

        self.proj = nn.Conv1d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  # Embedding dim으로 변환하며 패치크기의 커널로 패치크기만큼 이동하여 이미지를 패치로 분할 할 수 있음.

    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class FCGTransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention

    Returns:
        out: output of the encoder
    """

    def __init__(self,
                 signal_size=1024,
                 patch_size=16,
                 num_cls=17,
                 embed_dim=768,
                 num_layers=12,
                 ch=1,
                 n_heads=12,
                 expansion_factor=4,
                 p=0.):
        super(FCGTransformerEncoder, self).__init__()
        self.patch_embed = PatchEmbed(
            signal_size=signal_size,
            patch_size=patch_size,
            in_chans=ch,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.head = nn.Linear(embed_dim, num_cls)

    def forward(self, x):
        n_samples = x.shape[0]
        embed_out = self.patch_embed(x)  # (n_samples, n_patches, embed_dim)
        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        embed_out = torch.cat((cls_token, embed_out), dim=1)  # (n_samples, 1+n_patches, embed_dim)
        embed_out = embed_out + self.pos_embed  # (n_samples, 1+n_patches, embed_dim)
        embed_out = self.pos_drop(embed_out)

        for layer in self.layers:
            embed_out = layer(embed_out, embed_out, embed_out)  # (n_samples, 64, 768)

        embed_out = self.norm(embed_out)

        # cls_token_final = x[:, 0]  # just CLS token
        # x = self.head(cls_token_final)

        return embed_out


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads

        """
        self.attention = MultiHeadAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, value, x, mask):
        # Self-Attention
        self_attention = self.attention(x, x, x, mask=mask)  # 32x10x512
        query = self.dropout(self.norm(self_attention + x))
        # Cross-Attention
        out = self.transformer_block(key, query, value)

        return out


class ClassFormerDecoder(nn.Module):
    def __init__(self, embed_dim, num_layers=2, expansion_factor=4, n_heads=8, num_cls=1):
        super(ClassFormerDecoder, self).__init__()
        """  
        Args:
           target_vocab_size: vocabulary size of taget
           embed_dim: dimension of embedding
           seq_len : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention

        """
        self.query_emb = nn.Parameter(torch.randn(1, num_cls, embed_dim))
        # self.query_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=expansion_factor, n_heads=n_heads)
                for _ in range(num_layers)
            ]

        )
        # self.fc_out = nn.Linear(embed_dim*num_cls, num_cls)
        self.fc_out = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, enc_ebd):
        """
        :param enc_ebd:
        :return:
        """
        cls_emb = self.query_emb.expand(enc_ebd.size(0), -1, -1)

        for layer in self.layers:
            cls_emb = layer(enc_ebd, enc_ebd, cls_emb, mask=None)  # Key from Enc, Value from Enc, cls_emb

        # cls_emb = torch.flatten(cls_emb, -2, -1)
        out = self.fc_out(cls_emb)
        out = self.dropout(out)
        return torch.squeeze(out)


class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8, num_cls=1):
        super(TransformerDecoder, self).__init__()
        """  
        Args:
           target_vocab_size: vocabulary size of taget
           embed_dim: dimension of embedding
           seq_len : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention

        """
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=expansion_factor, n_heads=n_heads)
                for _ in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)
        # self.cls_emb = nn.Parameter(torch.randn(1, num_cls, embed_dim))

    def forward(self, x, enc_ebd, mask):
        """
        Args:
            x: input vector from target
            enc_ebd : output from encoder layer
            trg_mask: mask for decoder self attention
        Returns:
            out: output vector
        """

        x = self.word_embedding(x)  # 32x10x512
        x = self.position_embedding(x)  # 32x10x512
        x = self.dropout(x)

        # cls_emb = self.cls_emb.expand(enc_ebd.size(0), -1, -1)
        # enc_ebd = torch.cat((enc_ebd, cls_emb), 1)

        for layer in self.layers:
            x = layer(enc_ebd, enc_ebd, x, mask)  # Key from Enc, Value from Enc, x, Mask
            # x = layer(enc_ebd, x, enc_ebd, mask)  # Key from Enc, Value from Enc, x, Mask

        # out = F.softmax(self.fc_out(x), dim=2)
        out = self.fc_out(x)

        return out


class TransformerDecoderXport(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoderXport, self).__init__()
        """  
        Args:
           target_vocab_size: vocabulary size of taget
           embed_dim: dimension of embedding
           seq_len : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention

        """
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=4, n_heads=8)
                for _ in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_ebd, mask):
        """
        Args:
            x: input vector from target
            enc_ebd : output from encoder layer
            trg_mask: mask for decoder self attention
        Returns:
            out: output vector
        """

        x = self.word_embedding(x)  # 32x10x512
        x = self.position_embedding(x)  # 32x10x512
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(enc_ebd, enc_ebd, x, mask)  # Key from Enc, Value from Enc, x, Mask

        out = F.softmax(self.fc_out(x), dim=2)
        out = torch.squeeze(out, dim=0)
        return out[-1]


class FCGTransformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length, num_layers=2, expansion_factor=4,
                 n_heads=8):
        super(FCGTransformer, self).__init__()

        """  
        Args:
           embed_dim:  dimension of embedding 
           src_vocab_size: vocabulary size of source
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention

        """

        self.target_vocab_size = target_vocab_size

        self.encoder = FCGTransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers,
                                          expansion_factor=expansion_factor, n_heads=n_heads)
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers,
                                          expansion_factor=expansion_factor, n_heads=n_heads)

    def make_trg_mask(self, trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def prediction(self, src, bos_token):
        trg_mask = self.make_trg_mask(bos_token)
        enc_emb = self.encoder(src)
        out_labels = []
        batch_size, seq_len = src.shape[0], src.shape[1]
        shift_right = bos_token
        for i in range(seq_len):  # 10
            shift_right = self.decoder(shift_right, enc_emb, trg_mask)  # bs x seq_len x vocab_dim
            # taking the last token
            shift_right = shift_right[:, -1, :]

            shift_right = shift_right.argmax(-1)
            out_labels.append(shift_right.item())
            shift_right = torch.unsqueeze(shift_right, axis=0)

        return out_labels

    def forward(self, src, trg):
        """
        Args:
            src: input to encoder
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        trg_mask = self.make_trg_mask(trg)
        enc_emb = self.encoder(src)

        outputs = self.decoder(trg, enc_emb, trg_mask)
        return outputs


class FCGTransformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length, num_layers=2, expansion_factor=4,
                 n_heads=8):
        super(FCGTransformer, self).__init__()

        """  
        Args:
           embed_dim:  dimension of embedding 
           src_vocab_size: vocabulary size of source
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention

        """

        self.target_vocab_size = target_vocab_size

        self.encoder = FCGTransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers,
                                          expansion_factor=expansion_factor, n_heads=n_heads)

        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers,
                                          expansion_factor=expansion_factor, n_heads=n_heads)

    def make_trg_mask(self, trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def prediction(self, src, bos_token):
        trg_mask = self.make_trg_mask(bos_token)
        enc_emb = self.encoder(src)
        out_labels = []
        batch_size, seq_len = src.shape[0], src.shape[1]
        shift_right = bos_token
        for i in range(seq_len):  # 10
            shift_right = self.decoder(shift_right, enc_emb, trg_mask)  # bs x seq_len x vocab_dim
            # taking the last token
            shift_right = shift_right[:, -1, :]

            shift_right = shift_right.argmax(-1)
            out_labels.append(shift_right.item())
            shift_right = torch.unsqueeze(shift_right, axis=0)

        return out_labels

    def forward(self, src, trg):
        """
        Args:
            src: input to encoder
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        trg_mask = self.make_trg_mask(trg)
        enc_emb = self.encoder(src)

        outputs = self.decoder(trg, enc_emb, trg_mask)
        return outputs


if __name__ == '__main__':
    encoder = FCGTransformerEncoder(signal_size=1024, patch_size=16, n_classes=17, embed_dim=768, depth=12, n_heads=8, mlp_ratio=4)
    in_blob = torch.rand([1, 1, 1024])
    o = encoder(in_blob)

