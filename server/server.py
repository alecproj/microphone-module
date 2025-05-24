from __future__ import annotations

import argparse
import logging
import socket
import signal
import sys
import multiprocessing as mp
from typing import Optional, NoReturn, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, closing

import numpy as np
import numpy.typing as npt
import pyaudio
from rtp import RTP

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

# Type aliases
AudioArray = npt.NDArray[np.int16]
PacketData = bytes

def parse_args() -> Dict[str, Any]:
    """Parsing command line arguments"""
    parser = argparse.ArgumentParser(
        description="RTP Audio Server with Voice Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--disable-recognition",
        action="store_false",
        dest="enable_recognition",
        help="Disable voice recognition processing"
    )
    
    parser.add_argument(
        "--disable-playback",
        action="store_false",
        dest="enable_playback",
        help="Disable audio playback"
    )
    
    parser.add_argument(
        "--disable-file-write",
        action="store_false",
        dest="enable_file_write",
        help="Disable PCM file writing"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=Config.LOG_LEVEL,
        help="Set logging level"
    )
    
    return vars(parser.parse_args())

def configure_runtime(args: Dict[str, Any]) -> None:
    """Updating configuration based on arguments"""
    Config.ENABLE_RECOGNITION = args["enable_recognition"]
    Config.ENABLE_PLAYBACK = args["enable_playback"]
    Config.ENABLE_FILE_WRITE = args["enable_file_write"]
    Config.LOG_LEVEL = args["log_level"]

    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

logger = logging.getLogger("RTPServer")

class SpeechRecognizer(mp.Process):
    def __init__(
        self,
        audio_queue: mp.Queue,
        result_queue: mp.Queue,
        stop_event: mp.Event,
        model_path: str
    ):
        super().__init__()
        self.audio_queue = audio_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.model_path = model_path
        self.silence_timeout = 0.0

    def run(self) -> None:
        """Main cycle of the recognition process"""
        from vosk import Model, KaldiRecognizer
        import json
        
        model = Model(self.model_path)
        recognizer = KaldiRecognizer(model, Config.SAMPLE_RATE)

        self.result_queue.put(("log", "Recognizer started"))

        while not self.stop_event.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=0.5)
                
                if recognizer.AcceptWaveform(audio_data):
                    result = json.loads(recognizer.Result())
                    self.result_queue.put(("final", result.get("text", "")))

            except mp.queues.Empty:
                result = json.loads(recognizer.FinalResult())
                text = result.get('text', '')
                if text:
                    self.result_queue.put(("timeout", text))
                continue
            except Exception as e:
                self.result_queue.put(("error", str(e)))

class AudioStreamer:
    def __init__(self) -> None:
        # Playback
        self.p = pyaudio.PyAudio() if Config.ENABLE_PLAYBACK else None
        self.stream: Optional[pyaudio.Stream] = None

        # Recognition
        if Config.ENABLE_RECOGNITION:
            self.audio_queue = mp.Queue(maxsize=100)
            self.result_queue = mp.Queue()
            self.stop_event = mp.Event()
            self.recognizer_proc = SpeechRecognizer(
                self.audio_queue,
                self.result_queue,
                self.stop_event,
                Config.VOSK_MODEL_PATH
            )
        else:
            self.audio_queue = None
            self.result_queue = None
            self.stop_event = None
            self.recognizer_proc = None

        # File
        self.file_executor = ThreadPoolExecutor(max_workers=1) if Config.ENABLE_FILE_WRITE else None

    @contextmanager
    def audio_context(self) -> pyaudio.Stream:
        if not Config.ENABLE_PLAYBACK:
            yield None
            return
        try:
            stream = self.p.open(
                format=Config.AUDIO_FORMAT,
                channels=Config.CHANNELS,
                rate=Config.SAMPLE_RATE,
                output=True,
                frames_per_buffer=Config.FRAMES_PER_BUFFER
            )
            self.stream = stream
            yield stream
        finally:
            if stream and stream.is_active():
                stream.stop_stream()
                stream.close()
            self.p.terminate()
    
    def start_recognition(self):
        if Config.ENABLE_RECOGNITION:
            self.recognizer_proc.start()
            self.result_handler = ThreadPoolExecutor(max_workers=1)
            self.result_handler.submit(self._handle_results)
        else:
            logger.info("Voice recognition is disabled")

    def _handle_results(self):
        if not Config.ENABLE_RECOGNITION:
            return
        while not self.stop_event.is_set():
            try:
                res_type, text = self.result_queue.get(timeout=0.5)
                if res_type == "timeout":
                    print(f"\033[92mFinal (timeout): {text}\033[0m")
                elif res_type == "final":
                    print(f"\033[92mFinal: {text}\033[0m")
                elif res_type == "log":
                    logger.info(text)
                elif res_type == "error":
                    logger.error(f"Recognition error: {text}")
            except mp.queues.Empty:
                continue

    def process_audio(self, data: bytes):
        # Playback
        if Config.ENABLE_PLAYBACK and self.stream and self.stream.is_active():
            try:
                self.stream.write(data)
            except IOError as e:
                logger.error(f"Audio error: {str(e)}")

        # Recognition
        if Config.ENABLE_RECOGNITION and self.audio_queue:
            try:
                self.audio_queue.put_nowait(data)
            except mp.queues.Full:
                logger.warning("Audio queue full, dropping data")
        
        # Write file
        if Config.ENABLE_FILE_WRITE:
            self._async_write("TEST.PCM", data)

    def _async_write(self, filename: str, data: bytes) -> None:
        if not Config.ENABLE_FILE_WRITE or not self.file_executor:
            return

        def write_task():
            try:
                with open(filename, "ab") as f:
                    f.write(data)
            except IOError as e:
                logger.error(f"File error: {str(e)}")
        self.file_executor.submit(write_task)

    def shutdown(self):
        if Config.ENABLE_RECOGNITION:
            self.stop_event.set()
            self.recognizer_proc.join(timeout=5)
            if self.recognizer_proc.is_alive():
                self.recognizer_proc.terminate()
            self.result_handler.shutdown()

        if self.file_executor:
            self.file_executor.shutdown()

def validate_rtp_packet(packet: PacketData) -> bool:
    if len(packet) < Config.MIN_PACKET_LEN:
        logger.warning(f"Invalid packet length: {len(packet)} bytes")
        return False
    return True

def parse_rtp_payload(packet: PacketData) -> Optional[bytes]:
    try:
        rtp = RTP()
        rtp.fromBytearray(bytearray(packet))
        return rtp.payload if rtp.payload else None
    except Exception as e:
        logger.error(f"RTP parsing error: {str(e)}")
        return None

def pcm_to_array(pcm_data: bytes) -> Optional[AudioArray]:
    try:
        return np.frombuffer(pcm_data, dtype=np.int16)
    except ValueError as e:
        logger.error(f"PCM conversion error: {str(e)}")
        return None

class GracefulExiter:
    def __init__(self):
        self.running = True
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame) -> None:
        logger.info("Received termination signal")
        self.running = False

def udp_server() -> NoReturn:
    exiter = GracefulExiter()
    audio_streamer = AudioStreamer()
    audio_streamer.start_recognition()

    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
        sock.bind((Config.HOST, Config.PORT))
        sock.settimeout(1.0)  # Allow periodic check for exit signal

        with audio_streamer.audio_context():
            logger.info(f"Server started on {Config.HOST}:{Config.PORT} with settings:")
            logger.info(f" Recognition: {Config.ENABLE_RECOGNITION}")
            logger.info(f" Playback: {Config.ENABLE_PLAYBACK}")
            logger.info(f" File write: {Config.ENABLE_FILE_WRITE}")

            while exiter.running:
                try:
                    packet, addr = sock.recvfrom(Config.MAX_PACKET_LEN)
                    logger.debug(f"Received packet from {addr}")

                    if not validate_rtp_packet(packet):
                        continue

                    if (pcm_data := parse_rtp_payload(packet)) is None:
                        continue

                    if (audio_array := pcm_to_array(pcm_data)) is None:
                        continue

                    audio_streamer.process_audio(audio_array.tobytes())

                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    sys.exit(1)

    audio_streamer.shutdown()
    logger.info("Server shutdown complete")
    sys.exit(0)

if __name__ == "__main__":
    args = parse_args()
    configure_runtime(args)
    try:
        udp_server()
    except KeyboardInterrupt:
        logger.info("Manual interruption received")
        sys.exit(0)
