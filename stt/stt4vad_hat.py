import time
import numpy as np
import soundfile as sf
from hailo_platform.genai import Speech2Text, VDevice, Speech2TextTask

SAMPLE_RATE = 16000
# 이보다 짧은 클립은 오인식(영어 환청) 위험이 커서 STT를 건너뛴다.
MIN_AUDIO_SEC = 0.4
# 너무 짧은 발화는 무음으로 패딩해 인식을 안정화한다(whisper는 ~1s 이상에서 안정적).
PAD_TO_SEC = 1.0

# 무음/잡음에서 whisper가 흔히 지어내는 영어/무의미 환청들(소문자 비교).
_HALLUCINATIONS = {
    "you", "thank you", "thank you.", "thanks for watching",
    "thanks for watching!", "bye", "bye.", "고맙습니다", ".", "...",
}


class STTHandler:
    # whisper-small.hef는 language 파라미터를 무시하고 영어로 번역 출력하는
    # 알려진 버그가 있어(Hailo GenAI Model Zoo v5.2.0~5.3.0). base는 정상.
    def __init__(self, model_size=None, hef_path="/home/han/hailo_models/whisper-base.hef"):
        print("STT(HAT) 모델 로딩 중...")
        self.vdevice = VDevice()
        self.s2t = Speech2Text(self.vdevice, hef_path)
        print("STT(HAT) 모델 로딩 완료")

    def transcribe(self, audio_path):
        start_time = time.time()
        audio, sr = sf.read(audio_path, dtype="float32")

        # 스테레오로 들어오면 모노로 합친다.
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # whisper는 16kHz 전제. 다르면 경고(녹음부는 항상 16kHz로 저장).
        if sr != SAMPLE_RATE:
            print(f"⚠️  STT: 예상 샘플레이트 {SAMPLE_RATE}, 실제 {sr}")

        duration = len(audio) / sr if sr else 0.0
        # 너무 짧은 클립은 환청을 유발 → 아예 건너뛴다.
        if duration < MIN_AUDIO_SEC:
            print(f"⚠️  STT: 너무 짧은 발화({duration:.2f}s) — 건너뜀")
            return "", time.time() - start_time

        # 짧은 발화는 뒤에 무음을 붙여 길이를 확보한다.
        min_len = int(PAD_TO_SEC * sr)
        if len(audio) < min_len:
            audio = np.pad(audio, (0, min_len - len(audio)))

        text = self.s2t.generate_all_text(
            audio, task=Speech2TextTask.TRANSCRIBE, language="ko"
        ).strip()

        text = self._postprocess(text)
        return text, time.time() - start_time

    @staticmethod
    def _postprocess(text: str) -> str:
        """환청/영어 오인식을 걸러낸다. (한국어 대화 전제)"""
        if not text:
            return ""
        if text.lower().strip() in _HALLUCINATIONS:
            return ""
        # 한글이 한 글자도 없으면(영어로만 인식됨) 환청으로 간주하고 버린다.
        has_korean = any("가" <= ch <= "힣" for ch in text)
        if not has_korean:
            print(f"⚠️  STT: 한국어 아님(환청 추정) — 버림: {text!r}")
            return ""
        return text
