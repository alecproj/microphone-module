<div align="center">

# Audio Server
English | [Русский](./README.ru.md)
</div>

# Overview

The **Voice Audio Data Receiver Server** provides the following key features:
- 🌐 Receive audio stream over UDP (RTP protocol)
- 🔊 Real-time playback using PyAudio
- 💾 Record raw audio stream to PCM file
- 🎙️ Offline speech recognition via the [Vosk API](https://github.com/alphacep/vosk-api)
- ⚙️ Configuration through command-line arguments
- 🛡️ Proper handling of termination signals (SIGINT, SIGTERM)
### Dependencies

- Python 3.10+ (`python --version`)
- [Vosk API](https://alphacephei.com/vosk/)
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
- [NumPy](https://numpy.org/)
- [RTP library](https://github.com/bbc/rd-apmm-python-lib-rtp)
# Quick Start

## Installation

1. Clone the repository and switch to the server directory:
    ```bash
    git clone https://github.com/alecproj/microphone-module
    cd microphone-module/server
    ```
2. (Optional) Create a virtual environment:
    ```bash
    python -m venv venv
    # Linux/macOS:
    source venv/bin/activate
    # Windows:
    .\venv\Scripts\activate
    ```
3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Download and extract a Vosk model:
    ```bash
    wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
    unzip vosk-model-en-us-0.22.zip
    mv vosk-model-en-us-0.22 model
    ```
>[!TIP] 
Available models at [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)  

>[!NOTE] 
>If you don’t have `wget`, download and unpack manually into the `model` folder.

5. Update the model path in `server.py`:
    ```python
    VOSK_MODEL_PATH: str = "model/vosk-model-en-us-0.22"
    ```
## Usage

Run the server with:
```bash
python server.py [-h] [--disable-recognition] [--disable-playback] [--disable-file-write] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
```

**Command-line arguments:**

|Argument|Description|Default|
|---|---|---|
|-h, --help|Show help and exit|—|
|`--disable-recognition`|Disable speech recognition|Enabled|
|`--disable-playback`|Disable audio playback|Enabled|
|`--disable-file-write`|Disable writing audio to file (PCM)|Enabled|
|`--log-level`|Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)|`INFO`|

**Console output when server starts successfully:**
```sh
2025-06-26 16:49:28,664 - RTPServer - INFO - Server started on 0.0.0.0:8765 with settings:
2025-06-26 16:49:28,664 - RTPServer - INFO -  Recognition: True
2025-06-26 16:49:28,665 - RTPServer - INFO -  Playback: True
2025-06-26 16:49:28,665 - RTPServer - INFO -  File write: True
LOG (VoskAPI:ReadDataFiles():model.cc:213) Decoding params beam=13 max-active=7000 lattice-beam=6
LOG (VoskAPI:ReadDataFiles():model.cc:216) Silence phones 1:2:3:4:5:6:7:8:9:10
LOG (VoskAPI:RemoveOrphanNodes():nnet-nnet.cc:948) Removed 1 orphan nodes.
LOG (VoskAPI:RemoveOrphanComponents():nnet-nnet.cc:847) Removing 2 orphan components.
LOG (VoskAPI:Collapse():nnet-utils.cc:1488) Added 1 components, removed 2
LOG (VoskAPI:ReadDataFiles():model.cc:248) Loading i-vector extractor from model/vosk-model-ru-0.42/ivector/final.ie
LOG (VoskAPI:ComputeDerivedVars():ivector-extractor.cc:183) Computing derived variables for iVector extractor
LOG (VoskAPI:ComputeDerivedVars():ivector-extractor.cc:204) Done.
LOG (VoskAPI:ReadDataFiles():model.cc:279) Loading HCLG from model/vosk-model-ru-0.42/graph/HCLG.fst
LOG (VoskAPI:ReadDataFiles():model.cc:297) Loading words from model/vosk-model-ru-0.42/graph/words.txt
LOG (VoskAPI:ReadDataFiles():model.cc:308) Loading winfo model/vosk-model-ru-0.42/graph/phones/word_boundary.int
LOG (VoskAPI:ReadDataFiles():model.cc:315) Loading subtract G.fst model from model/vosk-model-ru-0.42/rescore/G.fst
LOG (VoskAPI:ReadDataFiles():model.cc:317) Loading CARPA model from model/vosk-model-ru-0.42/rescore/G.carpa
LOG (VoskAPI:ReadDataFiles():model.cc:323) Loading RNNLM model from model/vosk-model-ru-0.42/rnnlm/final.raw
2025-06-26 16:49:51,182 - RTPServer - INFO - Recognizer started
```
# For Developers

## Configuration

Main settings are defined in `server.py` (class `Config`), for example:
- `HOST` and `PORT` for the UDP server
- Audio parameters (format, channels, sample rate, buffer size)
- Vosk model path (`VOSK_MODEL_PATH`)
```python
class Config:
    HOST: str               = "0.0.0.0"
    PORT: int               = 8765
    AUDIO_FORMAT: int       = pyaudio.paInt16
    CHANNELS: int           = 1
    SAMPLE_RATE: int        = 16000
    FRAMES_PER_BUFFER: int  = 4096
    MIN_PACKET_LEN: int     = 512
    MAX_PACKET_LEN: int     = 1024
    LOG_LEVEL: str          = "INFO"
    VOSK_MODEL_PATH: str    = "model/vosk-model-ru-0.42"
    ENABLE_RECOGNITION: bool = True
    ENABLE_PLAYBACK: bool    = True
    ENABLE_FILE_WRITE: bool  = True
```
## Debugging Tips

1. Inspect incoming UDP traffic:
    ```bash
    tcpdump -i any udp port 8765 -vv -X
    ```
2. Bypass server playback using ffmpeg:
    ```bash
    ffmpeg -f alsa -i default -acodec pcm_s16le -ar 16000 -ac 1 -f rtp rtp://127.0.0.1:8765
    ```
3. Play back recorded PCM data:
    ```bash
    ffplay -f s16le -ar 16000 TEST.PCM
    ```
4. Simulate no network in `udp_server()`:
    ```python
    audio_streamer.process_audio(b"\x00" * 2048)  # generate silence
    ```
# TODO

- Add source validation
- Implement data encryption
- Monitor traffic and other metrics
- Write unit tests
