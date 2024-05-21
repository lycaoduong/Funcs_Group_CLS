import torch
import torch.nn as nn
from networks.OblabsFcg.FCGTransformer import FCGTransformerEncoder
from transformers import BertModel, BertTokenizer, AutoModel


class Textual(nn.Module):
    def __init__(self, model_dir='./chemical_bert', max_length=256):
        super(Textual, self).__init__()
        self.CBert = BertModel.from_pretrained(model_dir)
        self.Tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.max_length = max_length

    def __call__(self, text_input):
        bert_input = self.Tokenizer(text_input, padding='max_length', max_length=self.max_length, truncation=True,
                                    return_tensors="pt")
        id = bert_input['input_ids']
        mask = bert_input['attention_mask']
        _, x = self.CBert(input_ids=id, attention_mask=mask, return_dict=False)
        return x


class FcgClip(nn.Module):
    def __init__(self):
        super(FcgClip, self).__init__()
        self.Signalsual = FCGTransformerEncoder(signal_size=1024, patch_size=16, embed_dim=768,
                                             num_layers=2, n_heads=4, expansion_factor=2)
        self.Textual = Textual()
        self.logit_scale = 100.0

    def __call__(self, signal, text):
        x1 = self.Signalsual(signal)
        signal_features = x1[:, 0]
        text_features = self.Textual(text)

        # normalized features
        signal_features = signal_features / signal_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_signal = self.logit_scale * signal_features @ text_features.t()
        # logits_per_text = logits_per_signal.t()
        # Comment it if Loss Function has activation function
        # logits_per_signal = torch.sigmoid(logits_per_signal)
        return logits_per_signal


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


if __name__ == '__main__':
    model = FcgClip()
    signals = torch.rand(2, 1, 1024)
    texts = [["Signal contains ankane"], ["Signal contains Ankyn", "Signal contains Methyl"]]

    for idx, text in enumerate(texts):
        signal = torch.cat(signals[idx])
        logits_per_signal = model(signal, text)
        loss = clip_loss(logits_per_signal)
