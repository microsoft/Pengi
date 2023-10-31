import numpy as np
from transformers import AutoTokenizer
from models.pengi import PENGI
import os
import torch
from collections import OrderedDict
import librosa
from importlib_resources import files
import yaml
import argparse
import torchaudio
import torchaudio.transforms as T
import collections
import random

class PengiWrapper():
    """
    A class for interfacing Pengi model.
    """
    def __init__(self, config, use_cuda=False):
        self.file_path = os.path.realpath(__file__)
        if config == "base":
            config_path = 'base.yml'
            model_path = 'base.pth'
        elif config == "base_no_text_enc":
            config_path = 'base_no_text_enc.yml'
            model_path = 'base_no_text_enc.pth'
        else:
            raise ValueError(f"Config type {config} not supported")

        self.model_path = files('configs').joinpath(model_path)
        self.config_path = files('configs').joinpath(config_path)
        self.use_cuda = use_cuda
        self.model, self.enc_tokenizer, self.dec_tokenizer, self.args = self.get_model_and_tokenizer(config_path=self.config_path)
        self.model.eval()

    def read_config_as_args(self,config_path):
        return_dict = {}
        with open(config_path, "r") as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            return_dict[k] = v
        return argparse.Namespace(**return_dict)

    def get_model_and_tokenizer(self, config_path):
        r"""Load Pengi with args from config file"""
        args = self.read_config_as_args(config_path)
        args.prefix_dim = args.d_proj
        args.total_prefix_length = 2*args.prefix_length
        if not args.use_text_model:
            args.text_model = args.text_decoder

        # Copy relevant configs from dataset_config
        args.sampling_rate = args.dataset_config['sampling_rate']
        args.duration = args.dataset_config['duration']
        model = PENGI(
            # audio
            audioenc_name = args.audioenc_name,
            sample_rate = args.sampling_rate,
            window_size = args.window_size,
            hop_size = args.hop_size,
            mel_bins = args.mel_bins,
            fmin = args.fmin,
            fmax = args.fmax,
            classes_num = None,
            out_emb = args.out_emb,
            specaug = args.specaug,
            mixup = args.mixup,
            # text encoder
            use_text_encoder = args.use_text_model,
            text_encoder = args.text_model,
            text_encoder_embed_dim = args.transformer_embed_dim,
            freeze_text_encoder_weights = args.freeze_text_encoder_weights,
            # text decoder
            text_decoder = args.text_decoder,
            prefix_length = args.prefix_length,
            clip_length = args.prefix_length_clip,
            prefix_size = args.prefix_dim,
            num_layers = args.num_layers,
            normalize_prefix = args.normalize_prefix,
            mapping_type = args.mapping_type,
            freeze_text_decoder_weights = args.freeze_gpt_weights,
            # common
            d_proj = args.d_proj,
            use_pretrained_audioencoder = args.use_pretrained_audioencoder,
            freeze_audio_encoder_weights= args.freeze_audio_encoder_weights,
            use_precomputed_melspec = False,
            pretrained_audioencoder_path = None,
        )
        model.enc_text_len = args.dataset_config['enc_text_len']
        model.dec_text_len = args.dataset_config['dec_text_len']
        model_state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))['model']
        try:
            model.load_state_dict(model_state_dict)
        except:
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k[7:] # remove 'module.'
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        enc_tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        if 'gpt' in args.text_model:
            enc_tokenizer.add_special_tokens({'pad_token': '!'})

        dec_tokenizer = AutoTokenizer.from_pretrained(args.text_decoder)
        if 'gpt' in args.text_decoder:
            dec_tokenizer.add_special_tokens({'pad_token': '!'})

        if self.use_cuda and torch.cuda.is_available():
            model = model.cuda()
        
        return model, enc_tokenizer, dec_tokenizer, args

    def default_collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if self.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        self.default_collate_err_msg_format.format(elem.dtype))

                return self.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    'each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.default_collate(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))

    def load_audio_into_tensor(self, audio_path, audio_duration, resample=True):
        r"""Loads audio file and returns raw audio."""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        resample_rate = self.args.sampling_rate
        if resample and resample_rate != sample_rate:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
        audio_time_series = audio_time_series.reshape(-1)
        sample_rate = resample_rate

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
        if audio_duration*sample_rate >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                        audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:audio_duration*sample_rate]
        else:
            # audio_time_series is longer than predefined audio duration,
            # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - audio_duration*sample_rate)
            audio_time_series = audio_time_series[start_index:start_index +
                                                  audio_duration*sample_rate]
        return torch.FloatTensor(audio_time_series)

    def preprocess_audio(self, audio_files, resample):
        r"""Load list of audio files and return raw audio"""
        audio_tensors = []
        for audio_file in audio_files:
            audio_tensor = self.load_audio_into_tensor(
                audio_file, self.args.duration, resample)
            audio_tensor = audio_tensor.reshape(
                1, -1).cuda() if self.use_cuda and torch.cuda.is_available() else audio_tensor.reshape(1, -1)
            audio_tensors.append(audio_tensor)
        return self.default_collate(audio_tensors)

    def preprocess_text(self, prompts, enc_tok, add_text):
        r"""Load list of prompts and return tokenized text"""
        tokenized_texts = []
        tokenizer = self.enc_tokenizer if enc_tok else self.dec_tokenizer
        for ttext in prompts:
            if add_text:
                tok = self.dec_tokenizer.encode_plus(text=ttext, add_special_tokens=True, return_tensors="pt")
            else:
                if enc_tok:
                    ttext = ttext + ' <|endoftext|>' if 'gpt' in self.args.text_model else ttext
                tok = tokenizer.encode_plus(
                            text=ttext, add_special_tokens=True,\
                            max_length=self.model.enc_text_len, 
                            pad_to_max_length=True, return_tensors="pt")
                
            for key in tok.keys():
                tok[key] = tok[key].reshape(-1).cuda() if self.use_cuda and torch.cuda.is_available() else tok[key].reshape(-1)
            tokenized_texts.append(tok)
        return self.default_collate(tokenized_texts)
    
    def _get_audio_embeddings(self, preprocessed_audio):
        r"""Load preprocessed audio and return a audio embeddings"""
        with torch.no_grad():
            preprocessed_audio = preprocessed_audio.reshape(
                preprocessed_audio.shape[0], preprocessed_audio.shape[2])
            audio_embeddings = self.model.audio_encoder(preprocessed_audio)[0]
            if self.args.normalize_prefix:
                audio_embeddings = audio_embeddings / audio_embeddings.norm(2, -1).reshape(-1,1)
        return audio_embeddings

    def _get_audio_prefix(self, audio_embeddings):
        r"""Produces audio embedding which is fed to LM"""
        with torch.no_grad():
            audio_prefix = self.model.caption_decoder.audio_project(audio_embeddings).contiguous().view(-1, self.model.caption_decoder.prefix_length, self.model.caption_decoder.gpt_embedding_size)
        return audio_prefix

    def _get_prompts_embeddings(self, preprocessed_prompts):
        r"""Load preprocessed prompts and return a prompt embeddings"""
        with torch.no_grad():
            if self.args.use_text_model:
                prompts_embed = self.model.caption_encoder(preprocessed_prompts)
            else:
                prompts_embed = self.model.caption_decoder.gpt.transformer.wte(preprocessed_prompts['input_ids'])
        return prompts_embed
    
    def _get_prompts_prefix(self, prompts_embed):
        r"""Produces prompt prefix which is fed to LM"""
        with torch.no_grad():
            prompts_prefix = self.model.caption_decoder.text_project(prompts_embed).contiguous().view(-1, self.model.caption_decoder.prefix_length, self.model.caption_decoder.gpt_embedding_size)
        return prompts_prefix
    
    def _get_decoder_embeddings(self, preprocessed_text):
        r"""Load additional text and return a additional text embeddings"""
        with torch.no_grad():
            decoder_embed = self.model.caption_decoder.gpt.transformer.wte(preprocessed_text['input_ids'])
        return decoder_embed
    
    def _generate_beam(self, beam_size: int = 5, embed=None,
                  entry_length=67, temperature=1., stop_token: str = ' <|endoftext|>'):
        r"""Produces text conditioned embeddings using beam search"""
        stop_token_index = self.dec_tokenizer.encode(stop_token)[0]
        tokens = None
        scores = None
        device = next(self.model.parameters()).device
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        with torch.no_grad():
            generated = embed

            for i in range(entry_length):
                outputs = self.model.caption_decoder.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = self.model.caption_decoder.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [self.dec_tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]
        return output_texts, scores
    
    def get_audio_embeddings(self, audio_paths, resample=True):
        r"""Load list of audio files and return audio prefix and audio embeddings"""
        preprocessed_audio = self.preprocess_audio(audio_paths, resample)
        audio_embeddings = self._get_audio_embeddings(preprocessed_audio)
        audio_prefix = self._get_audio_prefix(audio_embeddings)
        return audio_prefix, audio_embeddings

    def get_prompt_embeddings(self, prompts):
        r"""Load list of text prompts and return prompt prefix and prompt embeddings"""
        preprocessed_text = self.preprocess_text(prompts, enc_tok=True, add_text=False)
        prompt_embeddings = self._get_prompts_embeddings(preprocessed_text)
        prompt_prefix = self._get_prompts_prefix(prompt_embeddings)
        return prompt_prefix, prompt_embeddings

    def get_decoder_embeddings(self, add_texts):
        r"""Load additional text and return a additional text embeddings"""
        preprocessed_text = self.preprocess_text(add_texts, enc_tok=False, add_text=True)
        addtext_embeddings = self._get_decoder_embeddings(preprocessed_text)
        return addtext_embeddings
    
    def generate(self,audio_paths, text_prompts, add_texts, max_len, beam_size, temperature, stop_token, audio_resample=True):
        r"""Produces text response for the given audio file and text prompts
        audio_paths: (list<str>) List of audio file paths
        text_prompts: (list<str>) List of text prompts corresponding to each audio in audio_paths. Refer to paper Table 1 and 11 for prompts and performance. 
                                  The default recommendation is to "generate metadata" prompt
        add_texts: (list<str>) List of additionl text or context corresponding to each audio in audio_paths
        max_len: (int) maximum length for text generation. Necessary to stop generation if GPT2 gets "stuck" producing same token
        beam_size: (int) beam size for beam search decoding. Beam size of 3 or 5 leads to reasonly performance-computation tradeoff
        temperature: (float) temperature parameter for GPT2 generation
        stop_token: (str) token used to stop text generation 
        audio_resample (bool) True for resampling audio. The model support only 44.1 kHz
        """
        if not isinstance(audio_paths, list):
            raise ValueError(f"The audio_paths is expected in list")
        if not isinstance(text_prompts, list):
            raise ValueError(f"The text_prompts is expected in list")
        if not isinstance(add_texts, list):
            raise ValueError(f"The add_texts is expected in list")
        length = len(audio_paths)
        if any(len(lst) != length for lst in [text_prompts, add_texts]):
            raise ValueError(f"The three inputs of audio, text and additional text should have same length")
        
        if stop_token is None:
            stop_token = ' <|endoftext|>'
        
        audio_prefix, _ = self.get_audio_embeddings(audio_paths, resample=audio_resample)
        prompt_prefix, _ = self.get_prompt_embeddings(text_prompts)
        
        preds = []
        for i in range(len(audio_paths)):
            if add_texts[i] == "" or add_texts[i] == None:
                prefix_embed = torch.cat([audio_prefix[i],prompt_prefix[i]],axis=0)
            else:
                add_embed = self.get_decoder_embeddings(add_texts[i])
                prefix_embed = torch.cat([audio_prefix[i],prompt_prefix[i],add_embed[i]],axis=0)
            prefix_embed = prefix_embed.unsqueeze(0)
            pred = self._generate_beam(embed=prefix_embed, beam_size=beam_size, temperature=temperature, stop_token=stop_token, entry_length=max_len)
            preds.append(pred)
        
        return preds
    
    def describe(self, audio_paths, max_len, beam_size, temperature, stop_token, audio_resample=True):
        r"""Produces text description using the given audio file and predefined text prompts
        audio_paths: (list<str>) List of audio file paths
        max_len: (int) maximum length for text generation. Necessary to stop generation if GPT2 gets "stuck" producing same token
        beam_size: (int) beam size for beam search decoding. Beam size of 3 or 5 leads to reasonly performance-computation tradeoff
        temperature: (float) temperature parameter for GPT2 generation
        stop_token: (str) token used to stop text generation 
        audio_resample (bool) True for resampling audio. The model support only 44.1 kHz
        """
        if not isinstance(audio_paths, list):
            raise ValueError(f"The audio_paths is expected in list")
        
        if stop_token is None:
            stop_token = ' <|endoftext|>'
        
        text_prompts = ["generate audio caption", "generate metadata", "this is a sound of"]
        audio_prefix, _ = self.get_audio_embeddings(audio_paths, resample=audio_resample)
        prompt_prefix, _ = self.get_prompt_embeddings(text_prompts)
        
        summaries = []
        for i in range(len(audio_paths)):
            preds = []
            for j in range(len(prompt_prefix)):
                prefix_embed = torch.cat([audio_prefix[i],prompt_prefix[j]],axis=0)
                prefix_embed = prefix_embed.unsqueeze(0)
                pred = self._generate_beam(embed=prefix_embed, beam_size=beam_size, temperature=temperature, stop_token=stop_token, entry_length=max_len)
                preds.append(pred[0][0])
            
            summary = preds[0] + preds[1] + 'this audio contains sound events: ' + preds[2][:-1] + '.'
            summaries.append(summary)
        
        return summaries
