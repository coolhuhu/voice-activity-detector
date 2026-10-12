import os
import unittest
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

from voice_activity_detector import (
    VoiceActivityDetector, 
    OnlineVoiceActivityDetector,
    SpeechSegmentStatus,
    VadType
)


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()
    

class TestVoiceActivityDetector(unittest.TestCase):    
    
    def offline_pipeline(self, vad: VoiceActivityDetector,
                         samples: np.ndarray, sample_rate: int):
        window_size = vad.window_size()
        print(f"vad.window_size: {window_size}")

        samples_without_silence = []
        offset = 0
        is_eof = False

        while not is_eof:
            if offset + window_size < samples.shape[0]:
                vad.accept_waveform(samples[offset:offset + window_size])
                offset += window_size
            else:
                vad.accept_waveform(samples[offset:])
                is_eof = True
                vad.flush()

            while not vad.is_empty():
                segment = vad.front()
                segment_start = segment.start
                segment_samples = segment.samples

                start_time = segment_start / sample_rate
                end_time = (segment_start + segment_samples.size) / sample_rate
                print(f"{start_time:.3f} -- {end_time:.3f}")

                samples_without_silence.append(segment_samples)
                vad.pop()

        print(f"Total speech segments: {len(samples_without_silence)}")
        print(f"Total samples without silence: {sum(len(s) for s in samples_without_silence)}")
        
    def vad_test_pipeline(self, model: str, wav_path: str, vad_type: VadType):
        samples, sample_rate = read_wave(wav_path)
        print(f"sample rate: {sample_rate}")

        threshold = 0.5
        min_silence_duration = 0.2
        min_speech_duration = 0.25
        max_speech_duration = 20

        if vad_type is VadType.Silero:
            vad = VoiceActivityDetector.init_silero(
                model,
                sample_rate,
                threshold,
                min_silence_duration,
                min_speech_duration,
                max_speech_duration)
        elif vad_type is VadType.Ten:
            vad = VoiceActivityDetector.init_ten(
                model,
                sample_rate,
                threshold,
                min_silence_duration,
                min_speech_duration,
                max_speech_duration)
        else:
            raise ValueError("Invalid VAD type")

        self.offline_pipeline(vad, samples, sample_rate)

    def test_ten_vad(self):
        print("test ten vad for 16KHz audio")
        
        current_dir = Path(__file__).resolve()
        model = os.path.join(current_dir, "../../../model/ten-vad.onnx")
        model = Path(model).resolve()
        print(f"ten vad model: {model}")

        wav_path = os.path.join(current_dir, "../../../tmp/vad-test.wav")
        wav_path = Path(wav_path).resolve()
        print(f"wav path: {wav_path}")

        self.vad_test_pipeline(str(model), str(wav_path), VadType.Ten)

    def test_silero_vad_16k(self):
        print("test silero_v6 vad for 16KHz audio")

        current_dir = Path(__file__).resolve()
        model = os.path.join(current_dir, "../../../model/silero_vad_v6.onnx")
        model = Path(model).resolve()
        print(f"silero vad model: {model}")

        wav_path = os.path.join(current_dir, "../../../tmp/vad-test.wav")
        wav_path = Path(wav_path).resolve()
        print(f"wav path: {wav_path}")

        self.vad_test_pipeline(str(model), str(wav_path), VadType.Silero)
    
    def test_silero_vad_8k(self):
        print("test silero_v6 vad for 8KHz audio")

        current_dir = Path(__file__).resolve()
        model = os.path.join(current_dir, "../../../model/silero_vad_v6.onnx")
        model = Path(model).resolve()
        print(f"silero vad model: {model}")

        wav_path = os.path.join(current_dir, "../../../tmp/vad-test-8k.wav")
        wav_path = Path(wav_path).resolve()
        print(f"wav path: {wav_path}")

        self.vad_test_pipeline(str(model), str(wav_path), VadType.Silero)
    
    def test_online_silero_vad(self):
        """Test OnlineVoiceActivityDetector with Silero VAD model."""
        print("test online silero vad for 16KHz audio")

        current_dir = Path(__file__).resolve()
        model = os.path.join(current_dir, "../../../model/silero_vad_v6.onnx")
        model = Path(model).resolve()
        print(f"silero vad model: {model}")

        wav_path = os.path.join(current_dir, "../../../tmp/vad-test.wav")
        wav_path = Path(wav_path).resolve()
        print(f"wav path: {wav_path}")

        samples, sample_rate = read_wave(str(wav_path))
        print(f"sample rate: {sample_rate}")
        print(f"total samples: {samples.shape[0]}")

        # Initialize VAD
        threshold = 0.5
        min_silence_duration = 0.2
        min_speech_duration = 0.25
        max_speech_duration = 20

        online_vad = OnlineVoiceActivityDetector(VadType.Silero, str(model), 
                                                sample_rate, threshold, 
                                                min_silence_duration, 
                                                min_speech_duration, 
                                                max_speech_duration)

        # Process audio in chunks (100ms)
        chunk_size = int(0.1 * sample_rate)
        offset = 0
        n = samples.shape[0]

        while offset + chunk_size < n:
            online_vad.accept_waveform(samples[offset: offset + chunk_size])
            offset += chunk_size

            if not online_vad.empty():
                segments = online_vad.get_speech_segments()
                for seg in segments:
                    if seg.start != -1:
                        start_time = seg.start / sample_rate
                        end_time = (seg.start + len(seg.samples)) / sample_rate
                        print(f"{start_time:.3f} -- {end_time:.3f} segment_id[{seg.segment_id}] status: {seg.status.name}")

                    if seg.status == SpeechSegmentStatus.kEND:
                        print("detect endpoint")
                        print()

        # Process remaining samples and flush
        online_vad.accept_waveform(samples[offset:])
        online_vad.flush()

        if not online_vad.empty():
            segments = online_vad.get_speech_segments()
            for seg in segments:
                if seg.start != -1:
                    start_time = seg.start / sample_rate
                    end_time = (seg.start + len(seg.samples)) / sample_rate
                    print(f"{start_time:.3f} -- {end_time:.3f} segment_id[{seg.segment_id}] status: {seg.status.name}")
                    
                if seg.status == SpeechSegmentStatus.kEND:
                    print("detect endpoint")


if __name__ == "__main__":
    unittest.main()