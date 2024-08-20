# CrisperWhisper++

**CrisperWhisper++** is an advanced variant of OpenAI's Whisper, designed for fast, precise, and verbatim speech recognition with accurate (**crisp**) word-level timestamps. Unlike the original Whisper, which tends to omit disfluencies and follows more of a intended transcription style, CrisperWhisper++ aims to transcribe every spoken word exactly as it is, including fillers, pauses, stutters and false starts.

## Key Features

- 🎯 **Accurate Word-Level Timestamps**: Provides precise timestamps, even around disfluencies and pauses, by utilizing an adjusted tokenizer and a custom attention loss during training.
- 📝 **Verbatim Transcription**: Transcribes every spoken word exactly as it is, including and differentiating fillers like "um" and "uh".
- 🔍 **Filler Detection**: Detects and accurately transcribes fillers.
- 🛡️ **Hallucination Mitigation**: Minimizes transcription hallucinations to enhance accuracy.

## Table of Contents

- [Key Features](#key-features)
- [Highlights](#highlights)
- [Performance Overview](#1-performance-overview)
  - [Qualitative Performance Overview](#11-qualitative-performance-overview)
  - [Quantitative Performance Overview](#12-quantitative-performance-overview)
    - [Transcription Performance](#transcription-performance)
    - [Segmentation Performance](#segmentation-performance)
- [Setup](#2-setup-⚙️)
  - [Prerequisites](#21-prerequisites)
  - [Environment Setup](#22-environment-setup)
- [Python Usage](#3-python-usage-🐍)
- [Running the Streamlit App](#4-running-the-streamlit-app)
    - [Prerequisites](#41-prerequisites)
    - [Steps to Run the Streamlit App](#42-steps-to-run-the-streamlit-app)
    - [Features of the App](#43-features-of-the-app)
- [How](#5-how)
- [License](#license)


## Highlights

- 🏆 **1st place** on the [OpenASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) in verbatim datasets (TED, AMI) and overall.
- 🎓 **Accepted at INTERSPEECH 2024**.
- 📄 **Paper Drop**: Check out our [ArXiv preprint](...) for details and reasoning behind our tokenizer adjustment.
- ✨ **New Feature**: Not mentioned in the paper is a added AttentionLoss to further improve timestamp accuracy. By specifically adding a loss to train the attention scores used for the DTW alignment using timestamped data we significantly boosted the alignment performance.



## 1. Performance Overview

### 1.1 Qualitative Performance Overview


| Audio | Whisper Large V3 | Crisper Whisper++ |
|-------|------------------------|------------------------|
| [Demo de 1](https://github.com/user-attachments/assets/c8608ca8-5e02-4c4a-afd3-8f7c5bff75d5) | Er war kein Genie, aber doch ein fehlender Ingegner. | Es ist zwar kein. Er ist zwar kein Genie, aber doch ein fähiger Ingenieur.|
| [Demo de 2](https://github.com/user-attachments/assets/c68414b1-0f84-441c-b39b-29069487edb6) | Leider müssen wir In diesen schweren Zeiten Auch unserem Tagesgeschäft nach gehen Der hier Vorgelegt Kultur Haushalt Der Ampel Regierung strebt An den Erfolg Der Union Zumindest Fortzuführen | Leider [UH] müssen wir in diesen [UH] schweren Zeiten auch [UH] unserem [UH] Tagesgeschäft nachgehen. Der hier [UH] vorgelegte [UH] Kulturhaushalt der [UH] Ampelregierung strebt an, den [UH] Erfolgskurs der Union [UH] zumindest [UH] fiskalisch fortzuführen. Es. |
| [Demo de 3](https://github.com/user-attachments/assets/0c1ed60c-2829-47e4-b7ba-eb584b0a5e9a) | die über alle FR-Fraktionen gut Im Blick behalten sollten, auch weil sie teilweise sehr teuer sind. Aber nicht nur, weil sie teuer sind. Wir steigen mit diesem Endentwurf ein, wir steigen mit diesem Endentwurf ein, wir steigen mit diesem Endentwurf ein, wir steigen mit diesem Endentwurf ein, wir steigen mit diesem Endentwurf ein, wir steigen mit diesem Endentwurf | Die über alle Fr Fraktionen hinweg gut im [UH] Blick behalten sollten, auch weil sie teil teilweise sehr te teuer sind. Aber nicht nur, weil sie te teuer sind. Wir [UH] steigen mit diesem Ent Entwurf ein in die sogenannten Pand Pandemiebereitschaftsverträge. |
| [Demo en 1](https://github.com/user-attachments/assets/cde5d69c-657f-4ae4-b4ae-b958ea2eacc5) | alternative, you can get those Dr. Bronner's | Alternative is you can get like [UH] you have those, you know, those doctor Brahmer's. |
| [Demo en 2](https://github.com/user-attachments/assets/906e307d-5613-4c41-9c61-65f4beede1fd) | influence natural surrounding? How does that influence the ecosystem? | Influence our [UM] our [UH] our natural surrounding. How does it influence our ecosystem? |
| [Demo en 3](https://github.com/user-attachments/assets/6c09cd58-a574-4697-9a7e-92e416cf2522) | And always find the place to park and you weren't long distance away from where you were trying to go. So,I remember that being fun and easy to do and there were nice places to go and good events to attend. Come downtown and you had the Warner Theater. | And always find a place on the street to park. And and it was it was easy and you weren't a long distance away from wherever it was that you were trying to go. So, I I I remember that being a lot of fun and easy to do and there were nice places to go and, [UM] i good events to attend. Come downtown and you had the Warner Theater and, [UM] |
| [Demo en 4](https://github.com/user-attachments/assets/7df19486-5e4e-4443-8528-09b07dddf61a) | you know, more masculine, who were rough, and that definitely wasn't me. Then, you know, that was my father's job because my father made sure that was smart, you know. So, you know, that hung around those people, you know, and then you had the one that were just just doing things that they shouldn't have been doing also. So, yeah, that was the little geek squad. | you know, more masculine, who were rough, and that definitely wasn't me. Then, you know, I was very smart because my father made sure I was smart. You know, so, [UM] you know, I I hung around those people, you know. And then you had the ones that were just just out doing things that they shouldn't have been doing also. So yeah, I was the l I was in the little geek squad. Do you |

### 1.2 Quantitative Performance Overview

#### Transcription Performance

CrisperWhisper++ significantly outperforms Whisper Large v3, especially on datasets that have a more verbatim transcription style in the ground truth, such as AMI and TED-LIUM.

| Dataset            | CrisperWhisper++ | Whisper Large v3 | 
|----------------------|:--------------:|:----------------:|
| [AMI](https://huggingface.co/datasets/edinburghcstr/ami)                 | **8.72**       | 16.01            |    
| [Earnings22](https://huggingface.co/datasets/revdotcom/earnings22)           | 12.37          | **11.3**        | 
| [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech)         | 10.27          | **10.02**        |     
| [LibriSpeech clean](https://huggingface.co/datasets/openslr/librispeech_asr)   | **1.74**       | 2.03            |    
| [LibriSpeech other](https://huggingface.co/datasets/openslr/librispeech_asr)   | 3.97           | **3.91**         |      
| [SPGISpeech](https://huggingface.co/datasets/kensho/spgispeech)          | **2.71**           | 2.95        |     
| [TED-LIUM](https://huggingface.co/datasets/LIUM/tedlium)             | **3.35**          | 3.9        |    
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli)           | **8.61**           | 9.52         |  
| [CommonVoice](https://huggingface.co/datasets/mozilla-foundation/common_voice_9_0)       | **8.19**           | 9.67        |      
| **Average WER**      | **6.66**       | 7.7         |  

#### Segmentation Performance

CrisperWhisper++ demonstrates superior performance in segmentation tasks, especially when measuring F1 Score and Avg IOU across multiple datasets.

| Dataset                          | Metric     | CrisperWhisper++ | Whisper Large v2 | Whisper Large v3 | WhisperTimestamped | WhisperX |
|----------------------------------|------------|------------------|------------------|------------------|--------------------|----------|
| [AMI IHM](https://groups.inf.ed.ac.uk/ami/corpus/)                           | F1 Score   | **0.90**         | 0.85             | 0.86             | 0.76               | 0.66     |
|                                  | Avg IOU    | **0.86**         | 0.74             | 0.77             | 0.75               | 0.60     |
| [Common Voice](https://commonvoice.mozilla.org/en/datasets)                           | F1 Score   | **0.82**         | 0.51             | 0.60             | 0.53               | 0.69     |
|                                  | Avg IOU    | **0.82**         | 0.74             | 0.76             | 0.73               | 0.64     |
| In-house Dataset (including pauses) | F1 Score   | **0.85**         | 0.57             | 0.69             | 0.43               | 0.73     |
|                                  | Avg IOU    | **0.74**         | 0.66             | 0.68             | 0.59               | 0.67     |
| [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)                            | F1 Score   | 0.80             | 0.67             | 0.72             | 0.68               | **0.83** |
|                                  | Avg IOU    | **0.83**         | 0.74             | 0.79             | 0.74               | 0.68     |

More plots and ablations can be found in the `run_experiments/plots` folder.

## 2. Setup ⚙️

### 2.1 Prerequisites

- **Python**: 3.10
- **PyTorch**: 2.0
- **NVIDIA Libraries**: cuBLAS 11.x and cuDNN 8.x (for GPU execution)

### 2.2 Environment Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/nyrahealth/crisperWhisper.git
    cd crisperWhisper
    ```

2. **Create Python Environment**:
    ```bash
    conda create --name crisperWhisper++ python=3.10
    conda activate crisperWhisper++
    ```


3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Additional Installations**:
    Follow OpenAI's instructions to install additional dependencies like `ffmpeg` and `rust`: [Whisper Setup](https://github.com/openai/whisper#setup).

## 3. Python Usage 🐍

Here's how to use CrisperWhisper++ in your Python scripts:


```python
import os
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "/home/azureuser/laurin/code/research/output/crisper_whisper_timestamp_finetuned"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]
result = pipe(sample, return_timestamps="word")
print(result)
```

## 4. Running the Streamlit App

To use the CrisperWhisper++ model with a user-friendly interface, you can run the provided Streamlit app. This app allows you to record or upload audio files for transcription and view the results with accurate word-level timestamps.

### 4.1 Prerequisites

Make sure you have followed the [Setup ⚙️](#setup) instructions above and have the `crisperWhisper++` environment activated.

### 4.2 Steps to Run the Streamlit App

1. **Activate the Conda Environment**

    Ensure you are in the `crisperWhisper++` environment:
    ```sh
    conda activate crisperWhisper++
    ```

2. **Navigate to the App Directory**

    Change directory to where the `app.py` script is located:


3. **Run the Streamlit App**

    Use the following command to run the app. Make sure to replace `/path/to/your/model` with the actual path to your CrisperWhisper++ model directory:
    ```sh
    streamlit run app.py -- --model_id /path/to/your/model
    ```

    For example:
    ```sh
    streamlit run app.py -- --model_id /home/azureuser/laurin/code/research/output/crisper_whisper++
    ```

4. **Access the App**

    After running the command, the Streamlit server will start, and you can access the app in your web browser at:
    ```
    http://localhost:8501
    ```

### 4.3 Features of the App

- **Record Audio**: Record audio directly using your microphone.
- **Upload Audio**: Upload audio files in formats like WAV, MP3, or OGG.
- **Transcription**: Get accurate verbatim transcriptions including fillers
- **Video Generation**: View the transcription with timestamps alongside a video with a black background.

## 5. How?


We employ the popular Dynamic Time Warping (DTW) on the Whisper cross-attention scores, as detailed in our [paper](...) to derive word-level timestamps. By leveraging our retokenization process, this method allows us to consistently detect pauses. Given that the accuracy of the timestamps heavily depends on the DTW cost matrix and, consequently, on the quality of the cross-attentions, we developed a specialized loss function for the selected alignment heads to enhance precision.

Although this loss function was not included in the original [paper](...) due to time constraints preventing the completion of experiments and training before the submission deadline, it has been used to train our publicly available models.
Key Features of this loss are as follows:

1. **Data Preparation**
    - We used datasets with word-level timestamp annotations, such as [AMI IHM](https://groups.inf.ed.ac.uk/ami/corpus/) and [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)   , but required additional timestamped data.
    - To address this, we validated the alignment accuracy of several forced alignment tools using a small hand-labeled dataset.
    - Based on this validation, we chose the [PyTorch CTC aligner](https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html) to generate more time-aligned data from the CommonVoice dataset.
    - Because the [PyTorch CTC aligner](https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html) tends to overestimate pause durations, we applied the same pause-splitting method detailed in our [paper](...) to correct these errors. The effectiveness of this correction was confirmed using our hand-labeled dataset.

2. **Token-Word Alignment**
    - Due to retokenization as detailed in our [paper](...), each token is either part of a word or a pause/space, but never both
    - Therefore each token can be cleanly aligned to a word OR a space/pause

3. **Ground Truth Cross-Attention**
    - We define the cross-attention ground truth for tokens as the L2-normalized vector, where:
        - A value of 1 indicates that the word is active according to the word-level ground truth timestamp.
        - A value of 0 indicates that no attention should be paid.
    - To account for small inaccuracies in the ground truth timestamps, we apply a linear interpolation of 4 steps (8 milliseconds) on both sides of the ground truth vector, transitioning smoothly from 0 to 1.

4. **Loss Calculation**
- The loss function is defined as `1 - cosine similarity`  between the predicted cross-attention vector (when predicting a token) and the ground truth cross-attention vector.
- This loss is averaged across all predicted tokens and alignment heads.


## License

[Specify the license under which this project is released]