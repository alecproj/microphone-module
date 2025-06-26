<div align="center">

# Аудио сервер
[English](./README.md) | Русский
</div>

# Обзор

**Сервер для приёма голосовых аудиоданных** обладает следующими ключевыми функциями:
- 🌐 Приём аудиопотока по UDP (протокол RTP)
- 🔊 Воспроизведение аудио в реальном времени с использованием PyAudio
- 💾 Запись сырого аудиопотока в файл формата PCM
- 🎙️ Офлайн-распознавание речи с помощью [Vosk API](https://github.com/alphacep/vosk-api)
- ⚙️ Конфигурация через аргументы командной строки
- 🛡️ Корректная обработка сигналов завершения (SIGINT, SIGTERM)

### Зависимости
- Python 3.10+ (проверьте версию: `python --version`)
- [VoskAPI](https://alphacephei.com/vosk/) 
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
- [NumPy](https://numpy.org)
- [RTP](https://github.com/bbc/rd-apmm-python-lib-rtp)
# Быстрый старт
## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/alecproj/microphone-module
cd microphone-module/server
```
2. (Опционально) Создайте виртуальное окружение:
```bash
python -m venv pyvenv
# Для Linux/macOS:
source pyvenv/bin/activate
# Для Windows:
.\pyvenv\Scripts\activate
```
3. Установите зависимости:
```bash
pip install -r requirements.txt
```
4. Установите модель Vosk:
```bash
# Замените модель при необходимости
wget https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip
unzip vosk-model-ru-0.42.zip
mv vosk-model-ru-0.42 model
```
> [!TIP] 
> Доступные модели: [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)

> [!NOTE]
> Если у вас нет `wget`, скачайте модель вручную и распакуйте в папку `model`.

5. Укажите путь к модели Vosk в файле [`server.py`](./server.py):
```python
    VOSK_MODEL_PATH: str = "model/vosk-model-ru-0.42"
```
## Использование

Запустите сервер, выполнив:
```bash
python server.py [-h] [--disable-recognition] [--disable-playback] [--disable-file-write] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
```
**Описание аргументов командной строки:**

| Аргумент                | Описание                                                    | По умолчанию |
| ----------------------- | ----------------------------------------------------------- | ------------ |
| -h, --help              | Помощь и завершение                                         | -            |
| `--disable-recognition` | Отключить распознавание речи                                | Включено     |
| `--disable-playback`    | Отключить воспроизведение аудио                             | Включено     |
| `--disable-file-write`  | Отключить запись аудио в файл (PCM)                         | Включено     |
| `--log-level`           | Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO         |

**Вывод в консоль при успешном запуске сервера:**
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
# Для разработчиков

## Конфигурация

 Основные параметры конфигурации изменяются в коде [`server.py`](./server.py) (класс `Config`), например:
- `HOST` и `PORT` для UDP сервера
- Параметры аудио (формат, каналы, частота дискретизации, размер буфера)
- Путь к модели Vosk (`VOSK_MODEL_PATH`)

```python
class Config:
    HOST: str = "0.0.0.0"
    PORT: int = 8765
    AUDIO_FORMAT: int = pyaudio.paInt16
    CHANNELS: int = 1
    SAMPLE_RATE: int = 16000
    FRAMES_PER_BUFFER: int = 4096 
    MIN_PACKET_LEN: int = 512
    MAX_PACKET_LEN: int = 1024
    LOG_LEVEL: str = "INFO"
    VOSK_MODEL_PATH: str = "model/vosk-model-ru-0.42"
    ENABLE_RECOGNITION: bool = True
    ENABLE_PLAYBACK: bool = True
    ENABLE_FILE_WRITE: bool = True

```
## Советы по отладке

1. Проверка входящего трафика:
```sh
tcpdump -i any udp port 8765 -vv -X
```
2. Прослушивание в обход сервера через ffmpeg
```sh
ffmpeg -f alsa -i default -acodec pcm_s16le -ar 16000 -ac 1 -f rtp rtp://127.0.0.1:8765
```
3. Воспроизведение записанного PCM:
```sh
ffplay -f s16le -ar 16000 TEST.PCM
```
4. Тестирование без сети:
```sh
# В udp_server() после получения данных:
audio_streamer.process_audio(b"\x00" * 2048)  # Генерация тишины
```
# TODO

-  Добавить валидацию источников
-  Внедрить шифрование данных
-  Реализовать мониторинг трафика и других метрик
-  Написать тесты
