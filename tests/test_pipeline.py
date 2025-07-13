# Run with:
# python -m tests.test_pipeline

from audio_utils.transcriber import Transcriber
from audio_utils.diarizer import Diarizer
from pipeline.audio_processing import build_speaker_segments
from text_utils.analyzer import TopicAnalyzer
from dotenv import load_dotenv
from datetime import datetime
import os
import json

def test_full_pipeline():
    pass