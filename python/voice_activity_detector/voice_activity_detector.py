"""
Voice Activity Detection (VAD) Python interface.

This module provides a high-level Python API for voice activity detection,
supporting Silero, TEN, and FireRed VAD models.

The VAD detects speech segments in audio streams and returns them as
SpeechSegment objects with timing information and audio samples.
"""

from voice_activity_detector.lib._voice_activity_detector import VadModelConfig
from voice_activity_detector.lib._voice_activity_detector import (
    VoiceActivityDetector as _VoiceActivityDetector,
)

from dataclasses import dataclass
from enum import Enum
from collections import deque
from typing import List

import numpy as np


class VadType(Enum):
    """Supported voice activity detection model types."""

    Silero = 1
    Ten = 2
    FireRed = 3


class VoiceActivityDetector:
    """
    Low-level voice activity detector wrapper.

    This class wraps the underlying C++ VAD implementation and provides
    a Python-friendly interface for processing audio waveforms and
    detecting speech segments.

    Attributes:
        sample_rate: Audio sample rate in Hz (default: 16000).
        num_threads: Number of threads for inference (default: 1).
        provider: Computation provider, e.g., 'cpu', 'cuda' (default: 'cpu').
        debug: Enable debug mode for additional logging (default: False).

    Example:
        >>> vad = VoiceActivityDetector.init_silero(
        ...     model='silero_vad.onnx',
        ...     sample_rate=16000,
        ...     threshold=0.5
        ... )
        >>> vad.accept_waveform(audio_samples)
        >>> if vad.is_speech_detected():
        ...     segment = vad.current_speech_segment()
        ...     print(f"Speech detected: {len(segment.samples)} samples")
    """

    sample_rate: int = 16000
    num_threads: int = 1
    provider: str = "cpu"
    debug: bool = False

    @classmethod
    def init_from_config(cls, config: VadModelConfig):
        """
        Initialize VAD from an existing configuration object.

        Args:
            config: Pre-configured VadModelConfig instance.

        Returns:
            A new VoiceActivityDetector instance.
        """
        if config.firered_vad.model and config.sample_rate != 16000:
            raise ValueError("FireRed VAD requires a 16000 Hz sample rate")
        if not config.Validate():
            raise ValueError("Invalid VAD configuration")

        self = cls.__new__(cls)
        self.config = config
        self.sample_rate = config.sample_rate
        self.num_threads = config.num_threads
        self.provider = config.provider
        self.debug = config.debug
        self.voice_activity_detector = _VoiceActivityDetector(self.config)

        return self

    @classmethod
    def init_silero(
        cls,
        model: str,
        sample_rate: int,
        threshold: float = 0.5,
        min_silence_duration: float = 0.2,
        min_speech_duration: float = 0.25,
        max_speech_duration: float = 20,
    ):
        """
        Initialize VAD with Silero model.

        Silero VAD is a lightweight and efficient voice activity detection
        model suitable for real-time applications.

        Args:
            model: Path to the Silero ONNX model file.
            sample_rate: Audio sample rate in Hz (16000 or 8000).
            threshold: Speech detection threshold (0.0 to 1.0). Default: 0.5.
            min_silence_duration: Minimum silence duration in seconds before
                marking end of speech. Default: 0.2.
            min_speech_duration: Minimum speech duration in seconds to
                consider as valid speech. Default: 0.25.
            max_speech_duration: Maximum speech duration in seconds before
                forcing end of speech. Default: 20.

        Returns:
            A new VoiceActivityDetector instance configured with Silero VAD.

        Raises:
            RuntimeError: If the configuration is invalid.
        """
        self = cls.__new__(cls)
        cls.sample_rate = sample_rate

        vad_config = VadModelConfig()
        vad_config.silero_vad.model = model
        vad_config.silero_vad.threshold = threshold
        vad_config.silero_vad.min_silence_duration = min_silence_duration
        vad_config.silero_vad.min_speech_duration = min_speech_duration
        vad_config.silero_vad.max_speech_duration = max_speech_duration
        vad_config.debug = cls.debug
        vad_config.sample_rate = cls.sample_rate

        # Window size depends on sample rate for optimal performance
        if cls.sample_rate == 16000:
            vad_config.silero_vad.window_size = 512
        else:
            vad_config.silero_vad.window_size = 256

        if not vad_config.Validate():
            raise RuntimeError("Invalid VAD config")

        self.config = vad_config
        self.voice_activity_detector = _VoiceActivityDetector(vad_config)

        return self

    @classmethod
    def init_ten(
        cls,
        model: str,
        sample_rate: int,
        threshold: float = 0.5,
        min_silence_duration: float = 0.5,
        min_speech_duration: float = 0.25,
        max_speech_duration: float = 20,
    ):
        """
        Initialize VAD with TEN model.

        Args:
            model: Path to the TEN ONNX model file.
            sample_rate: Audio sample rate in Hz.
            threshold: Speech detection threshold (0.0 to 1.0). Default: 0.5.
            min_silence_duration: Minimum silence duration in seconds before
                marking end of speech. Default: 0.5.
            min_speech_duration: Minimum speech duration in seconds to
                consider as valid speech. Default: 0.25.
            max_speech_duration: Maximum speech duration in seconds before
                forcing end of speech. Default: 20.

        Returns:
            A new VoiceActivityDetector instance configured with TEN VAD.

        Raises:
            RuntimeError: If the configuration is invalid.
        """
        self = cls.__new__(cls)
        cls.sample_rate = sample_rate

        vad_config = VadModelConfig()
        vad_config.ten_vad.model = model
        vad_config.ten_vad.threshold = threshold
        vad_config.ten_vad.min_silence_duration = min_silence_duration
        vad_config.ten_vad.min_speech_duration = min_speech_duration
        vad_config.ten_vad.max_speech_duration = max_speech_duration
        vad_config.debug = cls.debug
        vad_config.sample_rate = cls.sample_rate
        # Window size: 512 samples for 16kHz audio
        vad_config.ten_vad.window_size = 512

        if not vad_config.Validate():
            raise RuntimeError("Invalid VAD config")

        self.config = vad_config
        self.voice_activity_detector = _VoiceActivityDetector(vad_config)

        return self

    @classmethod
    def init_firered(
        cls,
        model: str,
        sample_rate: int,
        threshold: float = 0.5,
        min_silence_duration: float = 0.5,
        min_speech_duration: float = 0.25,
        max_speech_duration: float = 20,
    ):
        """Initialize FireRed VAD for 16 kHz audio with a 400-sample window.

        The model requires ``min_speech_duration >= 0.08`` seconds and
        ``0.01 <= threshold < 1``.
        """
        if sample_rate != 16000:
            raise ValueError("FireRed VAD requires a 16000 Hz sample rate")

        vad_config = VadModelConfig()
        vad_config.firered_vad.model = model
        vad_config.firered_vad.threshold = threshold
        vad_config.firered_vad.min_silence_duration = min_silence_duration
        vad_config.firered_vad.min_speech_duration = min_speech_duration
        vad_config.firered_vad.max_speech_duration = max_speech_duration
        vad_config.firered_vad.window_size = 400
        vad_config.sample_rate = sample_rate
        vad_config.num_threads = cls.num_threads
        vad_config.provider = cls.provider
        vad_config.debug = cls.debug

        if not vad_config.Validate():
            raise ValueError("Invalid FireRed VAD configuration")

        self = cls.__new__(cls)
        self.config = vad_config
        self.sample_rate = sample_rate
        self.num_threads = vad_config.num_threads
        self.provider = vad_config.provider
        self.debug = vad_config.debug
        self.voice_activity_detector = _VoiceActivityDetector(vad_config)
        return self

    def accept_waveform(self, samples: np.ndarray):
        """
        Submit audio samples for VAD processing.

        Args:
            samples: numpy array of float32 audio samples. The required window
                size depends on the selected model; use ``window_size()``.
        """
        self.voice_activity_detector.AcceptWaveform(samples)

    def is_empty(self) -> bool:
        """
        Check if the internal output queue is empty.

        Returns:
            True if no speech segments are pending output, False otherwise.
        """
        return self.voice_activity_detector.Empty()

    def pop(self):
        """
        Pop and return the oldest speech segment from the output queue.

        Returns:
            The next SpeechSegment in the queue, or None if empty.
        """
        return self.voice_activity_detector.Pop()

    def front(self):
        """
        Get the oldest speech segment without removing it from the queue.

        Returns:
            The oldest SpeechSegment, or None if queue is empty.
        """
        return self.voice_activity_detector.Front()

    def current_speech_segment(self):
        """
        Get the current speech segment being processed.

        Returns:
            The current SpeechSegment with latest detected speech.
        """
        return self.voice_activity_detector.CurrentSpeechSegment()

    def current_speech_segment_size(self) -> int:
        """
        Get the number of samples in the current speech segment.

        Returns:
            Number of samples in the current speech segment.
        """
        return self.voice_activity_detector.CurrentSpeechSegmentSize()

    def is_speech_detected(self) -> bool:
        """
        Check if speech is currently being detected.

        Returns:
            True if speech is detected in the current window, False otherwise.
        """
        return self.voice_activity_detector.IsSpeechDetected()

    def currentspeech_segment_start(self) -> int:
        """
        Get the start position of the current speech segment in samples.

        Returns:
            Start position of current speech segment relative to input stream.
        """
        return self.voice_activity_detector.CurrentSpeechSegmentStart()

    def window_size(self) -> int:
        """
        Get the VAD window size in samples.

        Returns:
            Window size in samples for the selected model.
        """
        return self.voice_activity_detector.WindowSize()

    def flush(self):
        """
        Flush remaining samples through the VAD.

        Call this when the audio stream ends to ensure all remaining
        samples are processed and any final speech segments are emitted.
        """
        self.voice_activity_detector.Flush()

    def reset(self):
        """
        Reset the VAD state.

        Clears all internal buffers and resets the detector to its
        initial state, ready for processing a new audio stream.
        """
        self.voice_activity_detector.Reset()


class SpeechSegmentStatus(Enum):
    """
    Status of a speech segment in the detection pipeline.

    Attributes:
        kBEG: Segment is the beginning of a speech utterance.
        kMIDDLE: Segment is a middle portion of a speech utterance.
        kEND: Segment is the end of a speech utterance.
        kINVALID: Segment is invalid or uninitialized.
    """

    kBEG = 0
    kMIDDLE = 1
    kEND = 2
    kINVALID = 3


@dataclass
class SpeechSegment:
    """
    Represents a detected speech segment.

    Contains the audio samples and metadata for a speech segment
    detected by the VAD.

    Attributes:
        start: Start position in samples relative to the input stream.
            -1 indicates an invalid or placeholder segment.
        samples: numpy array of float32 audio samples in this segment.
        segment_id: Unique identifier for the speech utterance this
            segment belongs to.
        status: SpeechSegmentStatus indicating position in the utterance.
    """

    start: int  # Start position in samples (-1 if invalid)
    samples: np.ndarray  # Audio samples as float32 numpy array
    segment_id: int  # Unique utterance identifier
    status: SpeechSegmentStatus  # Position in utterance

    def __init__(self):
        """Initialize an invalid SpeechSegment with default values."""
        self.start = -1
        self.samples = np.array([], dtype=np.float32)
        self.segment_id = 0
        self.status = SpeechSegmentStatus.kINVALID


class OnlineVoiceActivityDetector:
    """
    High-level online voice activity detector.

    This class provides a streaming interface for voice activity detection,
    automatically managing windowing, buffering, and segment assembly.
    It is designed for processing live audio streams with low latency.

    The detector processes audio in fixed-size windows and emits speech
    segments as they are completed, making it suitable for real-time
    applications like speech recognition pipelines.

    Attributes:
        vad_type: The type of VAD model being used.
        num_processed_samples: Total number of samples processed so far.
        speech_segment_start: Whether a speech segment is currently in progress.
        segment_id: Counter for assigning unique segment IDs.
        is_first_segment_in_current_speech: True if next segment is the
            start of a new speech utterance.
        speech_segments: Queue of detected speech segments awaiting retrieval.

    Example:
        >>> vad = OnlineVoiceActivityDetector(
        ...     vad_type=VadType.Silero,
        ...     model='silero_vad.onnx',
        ...     sample_rate=16000,
        ...     threshold=0.5
        ... )
        >>> # Process audio chunks as they arrive
        >>> vad.accept_waveform(audio_chunk)
        >>> segments = vad.get_speech_segments()
        >>> for seg in segments:
        ...     print(f"Segment {seg.segment_id}: {len(seg.samples)} samples")
        >>> # At end of stream
        >>> vad.flush()
        >>> segments = vad.get_speech_segments()
    """

    def __init__(
        self,
        vad_type: VadType,
        model: str,
        sample_rate: int,
        threshold: float = 0.5,
        min_silence_duration: float = 0.2,
        min_speech_duration: float = 0.25,
        max_speech_duration: float = 20,
    ):
        """
        Initialize the online VAD detector.

        Args:
            vad_type: Type of VAD model (Silero, Ten, or FireRed).
            model: Path to the ONNX model file.
            sample_rate: Audio sample rate in Hz.
            threshold: Speech detection threshold (0.0 to 1.0). Default: 0.5.
            min_silence_duration: Minimum silence duration in seconds before
                marking end of speech. Default: 0.2.
            min_speech_duration: Minimum speech duration in seconds to
                consider as valid speech. Default: 0.25.
            max_speech_duration: Maximum speech duration in seconds before
                forcing end of speech. Default: 20.

        Raises:
            ValueError: If vad_type is not a valid VadType enum value.
        """
        self._init_vad_detector(
            vad_type,
            model,
            sample_rate,
            threshold,
            min_silence_duration,
            min_speech_duration,
            max_speech_duration,
        )

        # Track processing state for segment assembly
        self.num_processed_samples = 0  # Total samples processed
        self.speech_segment_start = False  # Currently in a speech segment
        self.segment_id = -1  # Unique ID for each speech utterance
        self.is_first_segment_in_current_speech = False  # First segment flag
        self.speech_segments = deque()  # Output queue for speech segments

    def _init_vad_detector(
        self,
        vad_type: VadType,
        model: str,
        sample_rate: int,
        threshold: float = 0.5,
        min_silence_duration: float = 0.2,
        min_speech_duration: float = 0.25,
        max_speech_duration: float = 20,
    ):
        """
        Initialize the underlying VAD detector based on type.

        Args:
            vad_type: Type of VAD model to initialize.
            model: Path to the ONNX model file.
            sample_rate: Audio sample rate in Hz.
            threshold: Speech detection threshold.
            min_silence_duration: Minimum silence duration in seconds.
            min_speech_duration: Minimum speech duration in seconds.
            max_speech_duration: Maximum speech duration in seconds.

        Raises:
            ValueError: If vad_type is not recognized.
        """
        if vad_type is VadType.Silero:
            self.vad_detector = VoiceActivityDetector.init_silero(
                model,
                sample_rate,
                threshold,
                min_silence_duration,
                min_speech_duration,
                max_speech_duration,
            )
        elif vad_type is VadType.Ten:
            self.vad_detector = VoiceActivityDetector.init_ten(
                model,
                sample_rate,
                threshold,
                min_silence_duration,
                min_speech_duration,
                max_speech_duration,
            )
        elif vad_type is VadType.FireRed:
            self.vad_detector = VoiceActivityDetector.init_firered(
                model,
                sample_rate,
                threshold,
                min_silence_duration,
                min_speech_duration,
                max_speech_duration,
            )
        else:
            raise ValueError(
                "Invalid VAD type, vad_type should be one of "
                "VadType.Silero, VadType.Ten, or VadType.FireRed"
            )

    def accept_waveform(self, samples: np.ndarray):
        """
        Process incoming audio samples.

        Processes audio in two phases:
        1. Complete windows are processed and speech segments are emitted
           as they become available.
        2. Remaining samples (less than one window) are buffered for
           the next call or for flush at end of stream.

        Args:
            samples: numpy array of float32 audio samples.
        """
        n = len(samples)

        if n <= 0:
            return

        window_size = self.vad_detector.window_size()

        # Phase 1: Process complete windows
        offset = 0
        while offset + window_size < n:
            self.vad_detector.accept_waveform(samples[offset : offset + window_size])
            offset += window_size

            # Detect the start of a new speech segment
            if not self.speech_segment_start and self.vad_detector.is_speech_detected():
                self.speech_segment_start = True
                self.segment_id += 1
                self.is_first_segment_in_current_speech = True
                self.num_processed_samples = (
                    self.vad_detector.currentspeech_segment_start()
                )

            # Extract new speech segments from the detector
            if self.vad_detector.is_speech_detected():
                s = self.vad_detector.current_speech_segment()
                tail_size = s.start + len(s.samples) - self.num_processed_samples

                if tail_size > 0:
                    segment = SpeechSegment()
                    segment.start = self.num_processed_samples
                    segment.segment_id = self.segment_id
                    segment.samples = s.samples[-tail_size:].copy()
                    segment.status = (
                        SpeechSegmentStatus.kBEG
                        if self.is_first_segment_in_current_speech
                        else SpeechSegmentStatus.kMIDDLE
                    )

                    self.speech_segments.append(segment)
                    self.is_first_segment_in_current_speech = False
                    self.num_processed_samples = s.start + len(s.samples)

            # Drain completed segments from the detector's output queue
            while not self.vad_detector.is_empty():
                s = self.vad_detector.front()
                self.vad_detector.pop()

                # Check if there are remaining samples in this completed segment
                if self.num_processed_samples < (s.start + len(s.samples)):
                    tail_chunk_size = (
                        s.start + len(s.samples) - self.num_processed_samples
                    )

                    segment = SpeechSegment()
                    segment.start = self.num_processed_samples
                    segment.segment_id = self.segment_id
                    segment.samples = s.samples[-tail_chunk_size:].copy()
                    segment.status = SpeechSegmentStatus.kEND

                    self.speech_segments.append(segment)
                    self.num_processed_samples = s.start + len(s.samples)
                else:
                    # All samples have been output, but we still need to mark the endpoint
                    segment = SpeechSegment()
                    segment.start = -1
                    segment.segment_id = self.segment_id
                    segment.samples = np.array([], dtype=np.float32)
                    segment.status = SpeechSegmentStatus.kEND

                    self.speech_segments.append(segment)

                self.speech_segment_start = False

        # Phase 2: Handle remaining samples (< window_size)
        self.vad_detector.accept_waveform(samples[offset:])

        # If speech is detected in the remaining samples, handle segment start
        if self.vad_detector.is_speech_detected():
            if not self.speech_segment_start:
                self.speech_segment_start = True
                self.segment_id += 1
                self.is_first_segment_in_current_speech = True
                self.num_processed_samples = (
                    self.vad_detector.currentspeech_segment_start()
                )

            s = self.vad_detector.current_speech_segment()
            tail_size = s.start + len(s.samples) - self.num_processed_samples

            if tail_size > 0:
                segment = SpeechSegment()
                segment.start = self.num_processed_samples
                segment.segment_id = self.segment_id
                segment.samples = s.samples[-tail_size:].copy()
                segment.status = (
                    SpeechSegmentStatus.kBEG
                    if self.is_first_segment_in_current_speech
                    else SpeechSegmentStatus.kMIDDLE
                )

                self.speech_segments.append(segment)
                self.is_first_segment_in_current_speech = False
                self.num_processed_samples = s.start + len(s.samples)

    def empty(self) -> bool:
        """
        Check if there are any pending speech segments.

        Returns:
            True if no speech segments are pending, False otherwise.
        """
        return len(self.speech_segments) == 0

    def pop_all(self):
        """Clear all pending speech segments without returning them."""
        self.speech_segments.clear()

    def flush(self):
        """
        Flush the detector at end of audio stream.

        Should be called when the audio stream ends to ensure all
        remaining audio samples are properly processed and any
        final speech segments are emitted.

        After calling flush, call get_speech_segments() to retrieve
        any remaining segments.
        """
        # Flush the underlying VAD to process remaining samples
        self.vad_detector.flush()

        # Check if there's a pending speech segment that needs output
        if not self.vad_detector.is_empty():
            s = self.vad_detector.front()

            # Calculate the tail portion not yet output
            tail_size = s.start + len(s.samples) - self.num_processed_samples

            if tail_size > 0:
                segment = SpeechSegment()
                segment.start = self.num_processed_samples
                segment.segment_id = self.segment_id
                segment.samples = s.samples[-tail_size:].copy()
                segment.status = SpeechSegmentStatus.kEND

                self.speech_segments.append(segment)

    def get_speech_segments(self) -> List[SpeechSegment]:
        """
        Retrieve and clear all pending speech segments.

        Returns:
            List of SpeechSegment objects detected since last call.
            The internal segment queue is cleared after this call.

        Returns:
            An empty list if no segments are pending.
        """
        segments = []
        for seg in self.speech_segments:
            segments.append(seg)
        self.speech_segments.clear()
        return segments

    def reset(self):
        """
        Reset the detector state.

        Clears all internal buffers and counters, resetting the detector
        to its initial state. Call this before processing a new audio
        stream to avoid carrying over state from the previous stream.
        """
        self.vad_detector.reset()
        self.speech_segments.clear()
        self.num_processed_samples = 0
        self.speech_segment_start = False
        self.segment_id = -1
        self.is_first_segment_in_current_speech = False
