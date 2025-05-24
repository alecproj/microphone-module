from __future__ import annotations

import logging
import socket
import signal
import sys
from typing import Optional, Tuple, NoReturn
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, closing

import numpy as np
import numpy.typing as npt
import pyaudio
from rtp import RTP

# Configuration
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

# Type aliases
AudioArray = npt.NDArray[np.int16]
PacketData = bytes

# Configure logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RTPServer")

class AudioStreamer:
    def __init__(self) -> None:
        self.p = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = False

    @contextmanager
    def audio_context(self) -> pyaudio.Stream:
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

    def play_audio(self, data: bytes) -> None:
        if self.stream and self.stream.is_active():
            try:
                self.stream.write(data)
            except IOError as e:
                logger.error(f"Audio playback error: {str(e)}")

    def async_write(self, filename: str, data: bytes) -> None:
        def write_task():
            try:
                with open(filename, "ab") as f:
                    f.write(data)
            except IOError as e:
                logger.error(f"File write error: {str(e)}")
        
        self.executor.submit(write_task)

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

    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
        sock.bind((Config.HOST, Config.PORT))
        sock.settimeout(1.0)  # Allow periodic check for exit signal

        with audio_streamer.audio_context():
            logger.info(f"Server started on {Config.HOST}:{Config.PORT}")

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
                    
                    audio_streamer.play_audio(audio_array.tobytes())

                    # Async write to file
                    audio_streamer.async_write("TEST.PCM", audio_array.tobytes())

                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    sys.exit(1)

    logger.info("Server shutdown complete")
    sys.exit(0)

if __name__ == "__main__":
    try:
        udp_server()
    except KeyboardInterrupt:
        logger.info("Manual interruption received")
        sys.exit(0)
