import time
import soundfile as sf
from hailo_platform.genai import Speech2Text, VDevice, Speech2TextTask


class STTHandler:
    def __init__(self, model_size=None, hef_path="/home/han/hailo_models/whisper-small.hef"):
        print("STT(HAT) 모델 로딩 중...")
        self.vdevice = VDevice()
        self.s2t = Speech2Text(self.vdevice, hef_path)
        print("STT(HAT) 모델 로딩 완료")

    def transcribe(self, audio_path):
        start_time = time.time()
        audio, sr = sf.read(audio_path, dtype="float32")
        text = self.s2t.generate_all_text(
            audio, task=Speech2TextTask.TRANSCRIBE, language="ko"
        ).strip()
        elapsed_time = time.time() - start_time
        return text, elapsed_time
