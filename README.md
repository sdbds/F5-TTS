# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/SWivid/F5-TTS)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06885)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://swivid.github.io/F5-TTS/)
[![hfspace](https://img.shields.io/badge/🤗-Space%20demo-yellow)](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
[![msspace](https://img.shields.io/badge/🤖-Space%20demo-blue)](https://modelscope.cn/studios/modelscope/E2-F5-TTS)
[![lab](https://img.shields.io/badge/X--LANCE-Lab-grey?labelColor=lightgrey)](https://x-lance.sjtu.edu.cn/)
<img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto">

**F5-TTS**: Diffusion Transformer with ConvNeXt V2, faster trained and inference.

**E2 TTS**: Flat-UNet Transformer, closest reproduction from [paper](https://arxiv.org/abs/2406.18009).

**Sway Sampling**: Inference-time flow step sampling strategy, greatly improves performance

### Thanks to all the contributors !

## News
- **2024/10/08**: F5-TTS & E2 TTS base models on [🤗 Hugging Face](https://huggingface.co/SWivid/F5-TTS), [🤖 Model Scope](https://www.modelscope.cn/models/SWivid/F5-TTS_Emilia-ZH-EN), [🟣 Wisemodel](https://wisemodel.cn/models/SJTU_X-LANCE/F5-TTS_Emilia-ZH-EN).

## Installation

powershell run with `install-with-uv(nocache).ps1`

**[Optional]**: We provide [Dockerfile](https://github.com/SWivid/F5-TTS/blob/main/Dockerfile) and you can use the following command to build it.
```bash
# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n f5-tts python=3.10
conda activate f5-tts

# Install pytorch with your CUDA version, e.g.
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

Then you can choose from a few options below:

### 1. As a pip package (if just for inference)

```bash
pip install git+https://github.com/SWivid/F5-TTS.git
```

### 2. Local editable (if also do training, finetuning)

```bash
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
# git submodule update --init --recursive  # (optional, if need bigvgan)
pip install -e .
```

### 3. Docker usage
```bash
# Build from Dockerfile
docker build -t f5tts:v1 .

# Or pull from GitHub Container Registry
docker pull ghcr.io/swivid/f5-tts:main
```


## Inference

### 1. Gradio App

Currently supported features:

- Basic TTS with Chunk Inference
- Multi-Style / Multi-Speaker Generation
- Voice Chat powered by Qwen2.5-3B-Instruct
- [Custom inference with more language support](src/f5_tts/infer/SHARED.md)

```bash
# Launch a Gradio app (web interface)
f5-tts_infer-gradio

# Specify the port/host
f5-tts_infer-gradio --port 7860 --host 0.0.0.0

# Launch a share link
f5-tts_infer-gradio --share
```

### 2. CLI Inference

```bash
# Run with flags
# Leave --ref_text "" will have ASR model transcribe (extra GPU memory usage)
f5-tts_infer-cli \
--model "F5-TTS" \
--ref_audio "ref_audio.wav" \
--ref_text "The content, subtitle or transcription of reference audio." \
--gen_text "Some text you want TTS model generate for you."

# Run with default setting. src/f5_tts/infer/examples/basic/basic.toml
f5-tts_infer-cli
# Or with your own .toml file
f5-tts_infer-cli -c custom.toml

# Multi voice. See src/f5_tts/infer/README.md
f5-tts_infer-cli -c src/f5_tts/infer/examples/multi/story.toml
```

### 3. More instructions

- In order to have better generation results, take a moment to read [detailed guidance](src/f5_tts/infer).
- The [Issues](https://github.com/SWivid/F5-TTS/issues?q=is%3Aissue) are very useful, please try to find the solution by properly searching the keywords of problem encountered. If no answer found, then feel free to open an issue.


## Training

### 1. Gradio App

Read [training & finetuning guidance](src/f5_tts/train) for more instructions.

```bash
# Quick start with Gradio web interface
f5-tts_finetune-gradio
```


## [Evaluation](src/f5_tts/eval)


## Development

Use pre-commit to ensure code quality (will run linters and formatters automatically)

```bash
pip install pre-commit
pre-commit install
```

When making a pull request, before each commit, run: 

```bash
pre-commit run --all-files
```

Note: Some model components have linting exceptions for E722 to accommodate tensor notation


## Prepare Dataset

Example data processing scripts for Emilia and Wenetspeech4TTS, and you may tailor your own one along with a Dataset class in `model/dataset.py`.

```bash
# prepare custom dataset up to your need
# download corresponding dataset first, and fill in the path in scripts

# Prepare the Emilia dataset
python scripts/prepare_emilia.py

# Prepare the Wenetspeech4TTS dataset
python scripts/prepare_wenetspeech4tts.py
```

## Training & Finetuning

Once your datasets are prepared, you can start the training process.

```bash
# setup accelerate config, e.g. use multi-gpu ddp, fp16
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml     
accelerate config
accelerate launch train.py
```
An initial guidance on Finetuning [#57](https://github.com/SWivid/F5-TTS/discussions/57).

Gradio UI finetuning with `finetune_gradio.py` see [#143](https://github.com/SWivid/F5-TTS/discussions/143).

### Wandb Logging

By default, the training script does NOT use logging (assuming you didn't manually log in using `wandb login`).

To turn on wandb logging, you can either:

1. Manually login with `wandb login`: Learn more [here](https://docs.wandb.ai/ref/cli/wandb-login)
2. Automatically login programmatically by setting an environment variable: Get an API KEY at https://wandb.ai/site/ and set the environment variable as follows:

On Mac & Linux:

```
export WANDB_API_KEY=<YOUR WANDB API KEY>
```

On Windows:

```
set WANDB_API_KEY=<YOUR WANDB API KEY>
```
Moreover, if you couldn't access Wandb and want to log metrics offline, you can the environment variable as follows:

```
export WANDB_MODE=offline
```

## Inference

The pretrained model checkpoints can be reached at [🤗 Hugging Face](https://huggingface.co/SWivid/F5-TTS) and [🤖 Model Scope](https://www.modelscope.cn/models/SWivid/F5-TTS_Emilia-ZH-EN), or automatically downloaded with `inference-cli` and `gradio_app`.

Currently support 30s for a single generation, which is the **TOTAL** length of prompt audio and the generated. Batch inference with chunks is supported by `inference-cli` and `gradio_app`. 
- To avoid possible inference failures, make sure you have seen through the following instructions.
- A longer prompt audio allows shorter generated output. The part longer than 30s cannot be generated properly. Consider using a prompt audio <15s.
- Uppercased letters will be uttered letter by letter, so use lowercased letters for normal words. 
- Add some spaces (blank: " ") or punctuations (e.g. "," ".") to explicitly introduce some pauses. If first few words skipped in code-switched generation (cuz different speed with different languages), this might help.

### CLI Inference

Either you can specify everything in `inference-cli.toml` or override with flags. Leave `--ref_text ""` will have ASR model transcribe the reference audio automatically (use extra GPU memory). If encounter network error, consider use local ckpt, just set `ckpt_file` in `inference-cli.py`

for change model use `--ckpt_file` to specify the model you want to load,  
for change vocab.txt use `--vocab_file` to provide your vocab.txt file.

```bash
python inference-cli.py \
--model "F5-TTS" \
--ref_audio "tests/ref_audio/test_en_1_ref_short.wav" \
--ref_text "Some call me nature, others call me mother nature." \
--gen_text "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences."

python inference-cli.py \
--model "E2-TTS" \
--ref_audio "tests/ref_audio/test_zh_1_ref_short.wav" \
--ref_text "对，这就是我，万人敬仰的太乙真人。" \
--gen_text "突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道，我身上的肉，是为了掩饰我爆棚的魅力，否则，岂不吓坏了你们呢？"

# Multi voice
python inference-cli.py -c samples/story.toml
```

### Gradio App
Currently supported features:
- Chunk inference
- Podcast Generation
- Multiple Speech-Type Generation

You can launch a Gradio app (web interface) to launch a GUI for inference (will load ckpt from Huggingface, you may also use local file in `gradio_app.py`). Currently load ASR model, F5-TTS and E2 TTS all in once, thus use more GPU memory than `inference-cli`.

powershell run with `run_gui.ps1`

### Speech Editing

To test speech editing capabilities, use the following command.

```bash
python speech_edit.py
```

## Evaluation

### Prepare Test Datasets

1. Seed-TTS test set: Download from [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval).
2. LibriSpeech test-clean: Download from [OpenSLR](http://www.openslr.org/12/).
3. Unzip the downloaded datasets and place them in the data/ directory.
4. Update the path for the test-clean data in `scripts/eval_infer_batch.py`
5. Our filtered LibriSpeech-PC 4-10s subset is already under data/ in this repo

### Batch Inference for Test Set

To run batch inference for evaluations, execute the following commands:

```bash
# batch inference for evaluations
accelerate config  # if not set before
bash scripts/eval_infer_batch.sh
```

### Download Evaluation Model Checkpoints

1. Chinese ASR Model: [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh)
2. English ASR Model: [Faster-Whisper](https://huggingface.co/Systran/faster-whisper-large-v3)
3. WavLM Model: Download from [Google Drive](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view).

### Objective Evaluation

Install packages for evaluation:

```bash
pip install -r requirements_eval.txt
```

**Some Notes**

For faster-whisper with CUDA 11:

```bash
pip install --force-reinstall ctranslate2==3.24.0
```

(Recommended) To avoid possible ASR failures, such as abnormal repetitions in output:

```bash
pip install faster-whisper==0.10.1
```

Update the path with your batch-inferenced results, and carry out WER / SIM evaluations:
```bash
# Evaluation for Seed-TTS test set
python scripts/eval_seedtts_testset.py

# Evaluation for LibriSpeech-PC test-clean (cross-sentence)
python scripts/eval_librispeech_test_clean.py
```

## Acknowledgements

- [E2-TTS](https://arxiv.org/abs/2406.18009) brilliant work, simple and effective
- [Emilia](https://arxiv.org/abs/2407.05361), [WenetSpeech4TTS](https://arxiv.org/abs/2406.05763), [LibriTTS](https://arxiv.org/abs/1904.02882), [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) valuable datasets
- [lucidrains](https://github.com/lucidrains) initial CFM structure with also [bfs18](https://github.com/bfs18) for discussion
- [SD3](https://arxiv.org/abs/2403.03206) & [Hugging Face diffusers](https://github.com/huggingface/diffusers) DiT and MMDiT code structure
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) as ODE solver, [Vocos](https://huggingface.co/charactr/vocos-mel-24khz) and [BigVGAN](https://github.com/NVIDIA/BigVGAN) as vocoder
- [FunASR](https://github.com/modelscope/FunASR), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [UniSpeech](https://github.com/microsoft/UniSpeech), [SpeechMOS](https://github.com/tarepan/SpeechMOS) for evaluation tools
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) for speech edit test
- [mrfakename](https://x.com/realmrfakename) huggingface space demo ~
- [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx/tree/main) Implementation with MLX framework by [Lucas Newman](https://github.com/lucasnewman)
- [F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) ONNX Runtime version by [DakeQQ](https://github.com/DakeQQ)

## Citation
If our work and codebase is useful for you, please cite as:
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
## License

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia, which is an in-the-wild dataset. Sorry for any inconvenience this may cause.
