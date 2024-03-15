import numpy as np
import torch.nn as nn
import torch
from networks.OblabsFcg.FCGTransformer import TransformerDecoder, FCGTransformerEncoder, ClassFormerDecoder
import torch.nn.functional as F


# class FcgEngine_XPORT(nn.Module):
#     def __init__(self, embed_dim, img_size, patch_size, target_vocab_size, seq_length, num_layers=2, expansion_factor=4,
#                  n_heads=8):
#         super(FcgEngine_XPORT, self).__init__()
#         self.target_vocab_size = target_vocab_size
#
#         self.encoder = VisionTransformerEncoder(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
#                                                 depth=num_layers, n_heads=n_heads, mlp_ratio=expansion_factor)
#         self.decoder = TransformerDecoderXport(target_vocab_size, embed_dim, seq_length, num_layers=num_layers,
#                                           expansion_factor=expansion_factor, n_heads=n_heads)
#         # self.max_len = 12
#         # self.trg_mask = self.make_trg_mask(torch.tensor([[0]]))
#
#     def make_trg_mask(self, trg):
#         """
#         Args:
#             trg: target sequence
#         Returns:
#             trg_mask: target mask
#         """
#         batch_size, trg_len = trg.shape
#         # returns the lower triangular part of matrix filled with ones
#         trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
#             batch_size, 1, trg_len, trg_len
#         )
#         if torch.cuda.is_available():
#             trg_mask = trg_mask.to('cuda')
#         return trg_mask
#
#     def forward(self, img, tokenizer):
#         enc_emb = self.encoder(img)
#         trg_mask = self.make_trg_mask(tokenizer)
#         o = self.decoder(tokenizer, enc_emb[:, 1:, :], trg_mask)
#         return o

class FCGClassification(nn.Module):
    def __init__(self, embed_dim, signal_size, patch_size, num_layers=2, expansion_factor=4, n_heads=8, num_cls=1):
        super(FCGClassification, self).__init__()

        self.encoder = FCGTransformerEncoder(signal_size=signal_size, patch_size=patch_size, embed_dim=embed_dim,
                                             num_layers=num_layers, n_heads=n_heads, expansion_factor=expansion_factor,
                                             num_cls=num_cls)
        self.head = nn.Linear(embed_dim, num_cls)

    def forward(self, signal):
        x = self.encoder(signal)
        cls_token_final = x[:, 0]  # just CLS token
        x = self.head(cls_token_final)
        # Comment it if Loss Function has activation function
        # x = torch.sigmoid(x)
        return x


class FCGClassificationXPORT(nn.Module):
    def __init__(self, embed_dim, signal_size, patch_size, num_layers=2, expansion_factor=4, n_heads=8, num_cls=1):
        super(FCGClassificationXPORT, self).__init__()

        self.encoder = FCGTransformerEncoder(signal_size=signal_size, patch_size=patch_size, embed_dim=embed_dim,
                                             num_layers=num_layers, n_heads=n_heads, expansion_factor=expansion_factor,
                                             num_cls=num_cls)
        self.head = nn.Linear(embed_dim, num_cls)

    def get_self_attention(self, layer_value=None):
        layers = self.encoder.layers
        if layer_value is None:
            self_att = []
            for encoder_block in layers:
                self_att.append(encoder_block.attention.attention_map.cpu().numpy())
            return self_att
        # Extract layer value 0, 1 or 2 ...
        else:
            return layers[layer_value].attention.attention_map.cpu().numpy()


    def forward(self, signal):
        x = self.encoder(signal)
        cls_token_final = x[:, 0]  # just CLS token
        x = self.head(cls_token_final)
        # Comment it if Loss Function has activation function
        x = torch.sigmoid(x)
        return x


class FCGDescription(nn.Module):
    def __init__(self, embed_dim, signal_size, patch_size, target_vocab_size, seq_length, num_layers=2, expansion_factor=4,
                 n_heads=8, num_cls=1):
        super(FCGDescription, self).__init__()
        self.target_vocab_size = target_vocab_size

        self.encoder = FCGTransformerEncoder(signal_size=signal_size, patch_size=patch_size, embed_dim=embed_dim,
                                             num_layers=num_layers, n_heads=n_heads, expansion_factor=expansion_factor,
                                             num_cls=num_cls)
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers,
                                          expansion_factor=expansion_factor, n_heads=n_heads, num_cls=num_cls)
        self.seq_length = seq_length

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
        if torch.cuda.is_available():
            trg_mask = trg_mask.to('cuda')
        return trg_mask

    def prediction(self, signal):
        bos_token = [[1, 10]]
        enc_emb = self.encoder(signal)
        outputs = []
        for i in range(self.seq_length-1):
            shift_right = np.array(bos_token)
            trg_mask = self.make_trg_mask(shift_right)
            shift_right = torch.from_numpy(shift_right).to(torch.long)
            shift_right = shift_right.to('cuda')
            o = self.decoder(shift_right, enc_emb[:, 1:, :], trg_mask)
            pd = torch.softmax(o, dim=2)
            pd = pd[:, -1, :]
            pd = torch.argmax(pd, dim=-1).item()
            bos_token[0].append(pd)
            outputs.append(pd)
            if pd==2:
                break
        print(outputs)



    def forward(self, signal, trg):
        """
        Args:
            img: image to encoder
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        trg_mask = self.make_trg_mask(trg)
        enc_emb = self.encoder(signal)

        outputs = self.decoder(trg, enc_emb[:, 1:, :], trg_mask)
        return outputs


class FCGClassFormer(nn.Module):
    def __init__(self, embed_dim, signal_size, patch_size, num_layers=2, expansion_factor=4,
                 n_heads=8, num_cls=1):
        super(FCGClassFormer, self).__init__()

        self.encoder = FCGTransformerEncoder(signal_size=signal_size, patch_size=patch_size, embed_dim=embed_dim,
                                             num_layers=num_layers, n_heads=n_heads, expansion_factor=expansion_factor,
                                             num_cls=num_cls)
        self.decoder = ClassFormerDecoder(embed_dim=embed_dim, num_layers=num_layers, expansion_factor=expansion_factor,
                                          n_heads=n_heads, num_cls=num_cls)

    def forward(self, signal):
        """
        :param signal:
        :return:
        """
        enc_emb = self.encoder(signal)
        scores = self.decoder(enc_emb[:, 1:, :])
        # Comment it if Loss Function has activation function
        # scores = torch.sigmoid(scores)
        return scores


class FCGClassFormerXPORT(nn.Module):
    def __init__(self, embed_dim, signal_size, patch_size, target_vocab_size, num_layers=2, expansion_factor=4,
                 n_heads=8, num_cls=1):
        super(FCGClassFormerXPORT, self).__init__()
        self.target_vocab_size = target_vocab_size

        self.encoder = FCGTransformerEncoder(signal_size=signal_size, patch_size=patch_size, embed_dim=embed_dim,
                                             num_layers=num_layers, n_heads=n_heads, expansion_factor=expansion_factor,
                                             num_cls=num_cls)
        self.decoder = ClassFormerDecoder(embed_dim=embed_dim, num_layers=num_layers, expansion_factor=expansion_factor,
                                          n_heads=n_heads, num_cls=num_cls)

    def get_attention(self):
        layers = self.decoder.layers
        self_att = []
        cross_att = []
        for decoder_block in layers:
            self_att.append(decoder_block.attention.attention_map.cpu().numpy())
            cross_att.append(decoder_block.transformer_block.attention.attention_map.cpu().numpy())
        return {"self_att": self_att, "cross_att": cross_att}

    def forward(self, signal):
        """
        :param signal:
        :return:
        """
        enc_emb = self.encoder(signal)
        scores = self.decoder(enc_emb[:, 1:, :])
        scores = torch.sigmoid(scores)
        cross_att = self.decoder.layers[0].transformer_block.attention.attention_map
        return scores, cross_att


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    target_vocab_size = 20
    num_layers = 6
    seq_length = 8

    embed_dim = 768
    expansion_factor = 4
    n_heads = 8

    signal_size = 1024
    patch_size = 16

    # signal = torch.randn(2, 1, signal_size).to(device)
    # target = torch.tensor([[0, 1, 7, 4, 3, 5, 9, 2],
    #                        [0, 1, 5, 6, 2, 4, 7, 6]]).to(device)
    #
    # model = FCGDescription(embed_dim=embed_dim, signal_size=signal_size, patch_size=patch_size,
    #                          target_vocab_size=target_vocab_size, seq_length=seq_length, num_layers=num_layers,
    #                          expansion_factor=expansion_factor, n_heads=n_heads)
    #
    # model = model.to(device)
    #
    # o = model(signal, target)
    # print(o.shape)
    #
    # signal = torch.randn(1, 1, signal_size).to(device)
    # # target = torch.tensor([[0]]).to(device)
    # target = [0]
    #
    # o = model.prediction(signal, target, max_len=seq_length)
    # print(o)

    # Test Classification model
    # model = FCGClassification(embed_dim=embed_dim, signal_size=signal_size, patch_size=patch_size,
    #                        target_vocab_size=target_vocab_size, num_layers=num_layers,
    #                        expansion_factor=expansion_factor, n_heads=n_heads, num_cls=17)

    model = FCGClassFormer(embed_dim=embed_dim, signal_size=signal_size, patch_size=patch_size,
                           target_vocab_size=target_vocab_size, num_layers=num_layers,
                           expansion_factor=expansion_factor, n_heads=n_heads, num_cls=17)

    model.to(device)
    signal = torch.randn(2, 1, signal_size).to(device)
    o = model(signal)
    print(o)
    print(o.shape)
