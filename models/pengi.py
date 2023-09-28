import sys
sys.path.append('')
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel
import os

from models.audio import get_audio_encoder
from models.decoder import get_decoder

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        """Initialize a Batchnorm layer. """
        m.bias.data.fill_(0.)
        m.weight.data.fill_(1.)

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.linear1)
        init_layer(self.linear2)
        init_bn(self.layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name:str, d_in: int, d_out: int, sample_rate: int, window_size: int,
            hop_size: int, mel_bins: int, fmin: int, fmax: int, classes_num: int, 
            specaug: bool, mixup: bool, use_pretrained_audioencoder: bool, freeze_audio_encoder_weights: bool,
            use_precomputed_melspec: bool, pretrained_audioencoder_path: str) -> None:
        super().__init__()

        audio_encoder, pretrained_emb_size = get_audio_encoder(audioenc_name)

        if use_pretrained_audioencoder:
            classes_num = 527
            d_in = pretrained_emb_size

        self.base = audio_encoder(
            sample_rate, window_size,
            hop_size, mel_bins, fmin, fmax,
            classes_num, d_in,
            specaug, mixup, use_precomputed_melspec)

        self.projection = Projection(pretrained_emb_size if use_pretrained_audioencoder else d_in, d_out)

        if freeze_audio_encoder_weights:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output

class TextEncoder(nn.Module):
    def __init__(self, d_out: int, text_model: str, transformer_embed_dim: int, freeze_text_encoder_weights: bool) -> None:
        super().__init__()
        self.text_model = text_model
        self.base = AutoModel.from_pretrained(text_model)

        if 'clip' in text_model:
            self.clip_text_projection = self.base.text_projection
            self.base = self.base.text_model
            if 'base' in text_model:
                transformer_embed_dim = 512
        
        self.projection = Projection(transformer_embed_dim, d_out)

        if freeze_text_encoder_weights:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x):
        if 'clip' in self.text_model:
            pooled_output = self.base(**x)[1] # get pooled output
            out = self.clip_text_projection(pooled_output)  # get CLS token output
        else:
            out = self.base(**x)[0]
            out = out[:, 0, :]  # get CLS token output
        
        projected_vec = self.projection(out)
        return projected_vec

class PENGI(nn.Module):
    def __init__(self,
                # audio
                audioenc_name: str,
                sample_rate: int, 
                window_size: int, 
                hop_size: int, 
                mel_bins: int, 
                fmin: int, 
                fmax: int, 
                classes_num: int, 
                out_emb: int, 
                specaug: bool, 
                mixup: bool,
                # text encoder
                use_text_encoder: bool,
                text_encoder: str,
                text_encoder_embed_dim: int,
                freeze_text_encoder_weights: bool,
                # text decoder
                text_decoder: str,
                prefix_length: int,
                clip_length: int,
                prefix_size: int,
                num_layers: int,
                normalize_prefix: bool,
                mapping_type: str,
                freeze_text_decoder_weights: bool,
                # common
                d_proj: int,
                use_pretrained_audioencoder: bool,
                freeze_audio_encoder_weights: bool,
                use_precomputed_melspec: bool = False,
                pretrained_audioencoder_path: str = None,
                ):
        super().__init__()
        
        self.audio_encoder = AudioEncoder(
            audioenc_name, out_emb, d_proj,
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, 
            specaug, mixup, use_pretrained_audioencoder, freeze_audio_encoder_weights,
            use_precomputed_melspec, pretrained_audioencoder_path)

        self.use_text_encoder = use_text_encoder
        if self.use_text_encoder:
            self.caption_encoder = TextEncoder(
                d_proj, 
                text_encoder, text_encoder_embed_dim,
                freeze_text_encoder_weights
            )

        self.caption_decoder = get_decoder('Decoder')(
            text_decoder, prefix_length, clip_length, prefix_size,
            num_layers, normalize_prefix, mapping_type, freeze_text_decoder_weights,
            use_text_encoder,
        )

    def forward(self, audio, texts_enc, texts_dec):
        audio_embed, _ = self.audio_encoder(audio)
        if self.use_text_encoder:
            caption_embed = self.caption_encoder(texts_enc)
        else:
            caption_embed = self.caption_decoder.gpt.transformer.wte(texts_enc['input_ids'])

        out = self.caption_decoder(audio_embed, caption_embed, texts_dec)
        return out
    
    def generate_prefix_inference(self, audio, texts_enc):
        audio_embed, _ = self.audio_encoder(audio)
        if self.use_text_encoder:
            caption_embed = self.caption_encoder(texts_enc)
        else:
            caption_embed = self.caption_decoder.gpt.transformer.wte(texts_enc['input_ids'])
        prefix = self.caption_decoder.generate_prefix_inference(audio_embed, caption_embed)
        return prefix