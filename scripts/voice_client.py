"""Terminal voice client: Live VAD-based recording, interruptible playback."""

import sys
import tempfile
import threading
import time
from pathlib import Path
import numpy as np
import httpx
import sounddevice as sd
import soundfile as sf

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_URL = "http://localhost:8000"
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.02      # RMS amplitude threshold to trigger recording
SILENCE_DURATION = 1.5    # Seconds of silence to stop recording

class LiveVoiceClient:
    def __init__(self):
        self.is_recording = False
        self.is_playing = False
        self.audio_frames = []
        self.silence_start = None
        
        self.playback_event = threading.Event()
        self.playback_thread = None

    def audio_callback(self, indata, frames, time_info, status):
        """Monitor microphone input continuously."""
        volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
        
        if self.is_playing:
            # Barge-in detection
            if volume_norm > VAD_THRESHOLD * 2:  # Higher threshold to interrupt
                print("\n[Interrupt] Stopping playback...")
                self.playback_event.set()
                sd.stop()
                self.is_playing = False
            return

        if volume_norm > VAD_THRESHOLD:
            if not self.is_recording:
                print("\n[Listening...] Speaking detected.")
                self.is_recording = True
                self.audio_frames = []
            self.silence_start = None
        else:
            if self.is_recording:
                if self.silence_start is None:
                    self.silence_start = time.time()
                elif time.time() - self.silence_start > SILENCE_DURATION:
                    self.is_recording = False
                    self.process_audio()

        if self.is_recording:
            self.audio_frames.append(indata.copy())

    def process_audio(self):
        """Send the recorded audio to the API."""
        if len(self.audio_frames) == 0:
            return
            
        print("[Processing] Sending to API...")
        audio_data = np.concatenate(self.audio_frames, axis=0)
        self.audio_frames = []
        
        # Save to temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, SAMPLE_RATE)
            wav_bytes = Path(f.name).read_bytes()
        
        if len(wav_bytes) < 1000:
            print("[Processing] Audio too short, ignoring.")
            return

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{API_URL}/api/voice-query/audio",
                    files={"audio": ("recording.wav", wav_bytes, "audio/wav")},
                )
            resp.raise_for_status()
            self.play_audio(resp.content)
        except Exception as e:
            print(f"[Error] API failed: {e}")

    def play_audio(self, wav_bytes: bytes):
        """Play the AI response audio. Can be interrupted."""
        self.is_playing = True
        self.playback_event.clear()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            path = f.name
            
        try:
            data, rate = sf.read(path)
            print("🔊 [Speaking...] Mike is responding.")
            
            # Start non-blocking playback
            sd.play(data, rate)
            
            # Wait until finished or interrupted
            while sd.wait(100) and self.is_playing:
                if self.playback_event.is_set():
                    break
        finally:
            self.is_playing = False
            sd.stop()
            Path(path).unlink(missing_ok=True)
            print("🎙️ [Ready] Start speaking anytime...")

    def run(self):
        print("=" * 60)
        print("Mike - Interactive Voice Assistant")
        print("=" * 60)
        print("Features: Hands-free Voice Activity Detection (VAD) & Barge-in interrupts")
        print("Speak naturally. The bot will listen when you talk and process when you pause.")
        print("Press Ctrl+C to exit.\n")
        print("🎙️ [Ready] Start speaking anytime...")
        
        try:
            with sd.InputStream(callback=self.audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=4096):
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")

if __name__ == "__main__":
    client = LiveVoiceClient()
    client.run()
