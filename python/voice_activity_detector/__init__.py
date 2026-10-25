"""voice_activity_detector - Voice Activity Detection Python binding

This package provides Python bindings for the voice-activity-detector
C++ library using pybind11.
"""

from .voice_activity_detector import (
    VoiceActivityDetector,
    OnlineVoiceActivityDetector,
    SpeechSegment,
    SpeechSegmentStatus,
    VadType,
)
