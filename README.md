# 🐧Pengi: An Audio Language Model for Audio Tasks
[[`Paper`](https://arxiv.org/abs/2305.11834)] [[`Checkpoints`](https://zenodo.org/record/8387083)]

Pengi is an Audio Language Model that leverages Transfer Learning by framing all audio tasks as text-generation tasks. It takes as input, an audio recording, and text, and generates free-form text as output. The unified architecture of Pengi enables open-ended tasks and close-ended tasks without any additional fine-tuning or task-specific extensions.
![image](https://github.com/microsoft/Pengi/assets/28994673/abc714fb-cee3-4253-a753-0db4bd122144)

## News
[Sep 23] 🐧Pengi is accepted at NeurIPS 2023

## Setup
1. You are required to install the dependencies: `pip install -r requirements.txt`. If you have [conda](https://www.anaconda.com) installed, you can run the following: 

```shell
cd Pengi && \
conda create -n pengi python=3.8 && \
conda activate pengi && \
pip install -r requirements.txt
```

2. Download Pengi weights: [Pretrained Model \[Zenodo\]](https://zenodo.org/record/8387083)
3. Move the `base.pth` and `base_no_text_enc.pth` under `configs` folder

## Supported models
![pengi_1-main_arch](https://github.com/soham97/Pengi_api_review/assets/28994673/f2f36fb1-1c43-481c-906b-bd309b586b07)
The wrapper supports two models. The `base` option is Pengi architecture reported in paper and shown above. The `base_no_text_enc` is the Pengi architecture without the text encoder and only $m_2$ to encode tokenized text. All models only support 44.1 kHz input audio.

## Usage
The wrapper provides an easy way to get Pengi output given and audio and text input. To use the wrapper, inputs required are:
- `config`: Choose between "base" or "base_no_text_enc"
- `audio_file_paths`: List of audio file paths for inference 
- `text_prompts`: List of input text prompts corresponding to each of the files in audio_file_paths. Example: ["generate metadata", "generate metadata"]. Refer to Table 1 and 11 for prompts and performance in [paper](https://arxiv.org/pdf/2305.11834.pdf). The default recommendation is to "generate metadata" prompt
- `add_texts`: List of additional text corresponding to each of the files in audio_file_paths and prompt in prompts. This is used additional text input user can provide to guide GPT2.

Supported functions:
- `generate`: Produces text response for the given audio file and text prompts
- `describe`: Produces text description of the given audio file by concatenating the concatenating output of predefined text prompts
- `get_audio_embeddings`: Load list of audio files and return audio prefix and audio embeddings
- `get_prompt_embeddings`: Load list of text prompts and return prompt prefix and embeddings

### Text generation
```python
from wrapper import PengiWrapper as Pengi

pengi = Pengi(config="<choice of config>")

generated_response = pengi.generate(audio_paths=audio_file_paths,
                                    text_prompts=["generate metadata"], 
                                    add_texts=[""], 
                                    max_len=30, 
                                    beam_size=3, 
                                    temperature=1.0, 
                                    stop_token=' <|endoftext|>'
                                    )
```

### Audio description
```python
from wrapper import PengiWrapper as Pengi

pengi = Pengi(config="<choice of config>")

generated_summary = pengi.describe(audio_paths=audio_file_paths,
                                    max_len=30, 
                                    beam_size=3, 
                                    temperature=1.0, 
                                    stop_token=' <|endoftext|>'
                                    )
```

### Generate audio, audio prefix and prompt embeddings
```python
audio_prefix, audio_embeddings = pengi.get_audio_embeddings(audio_paths=audio_file_paths)

text_prefix, text_embeddings = pengi.get_prompt_embeddings(prompts=["generate metadata"])
```

## Citation
```BibTeX
@inproceedings{Pengi,
  title={Pengi: An Audio Language Model for Audio Tasks},
  author={Soham Deshmukh and Benjamin Elizalde and Rita Singh and Huaming Wang},
  journal={arXiv preprint arXiv:2305.11834},
  year={2023}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
