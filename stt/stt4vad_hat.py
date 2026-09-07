import time
import re
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

        # 발화 앞뒤의 무음/저에너지 구간을 잘라낸다.
        # whisper는 뒤에 붙은 무음에서 같은 말을 반복하는 환청("바지에 바지에…")을
        # 잘 내는데, 그 무음을 애초에 넣지 않으면 반복이 크게 줄어든다.
        audio = self._trim_silence(audio, sr)

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
    def _trim_silence(audio, sr, frame_ms=30, thresh_ratio=0.12, pad_ms=150):
        """피크 대비 저에너지인 앞/뒤 구간을 제거한다(무음 → 반복 환청 방지).

        프레임별 RMS를 구해, 피크의 thresh_ratio 밑으로 떨어지는 앞뒤 구간을 자른다.
        말이 시작/끝나는 지점 바깥으로 pad_ms 만큼은 여유로 남긴다.
        """
        frame = max(1, int(sr * frame_ms / 1000))
        n = len(audio) // frame
        if n < 2:
            return audio
        frames = audio[:n * frame].reshape(n, frame)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
        peak = float(rms.max())
        if peak <= 0:
            return audio
        active = np.where(rms > peak * thresh_ratio)[0]
        if len(active) == 0:
            return audio
        pad = frame * max(1, int(pad_ms / frame_ms))
        start = max(0, active[0] * frame - pad)
        end = min(len(audio), (active[-1] + 1) * frame + pad)
        return audio[start:end]

    @classmethod
    def _postprocess(cls, text: str) -> str:
        """환청/영어 오인식/반복을 걸러낸다. (한국어 대화 전제)"""
        if not text:
            return ""
        # 반복 환청 접기: "바지에 바지에 바지에" → "바지에"
        text = cls._collapse_repeats(text)
        if text.lower().strip() in _HALLUCINATIONS:
            return ""
        # 한글이 한 글자도 없으면(영어로만 인식됨) 환청으로 간주하고 버린다.
        has_korean = any("가" <= ch <= "힣" for ch in text)
        if not has_korean:
            print(f"⚠️  STT: 한국어 아님(환청 추정) — 버림: {text!r}")
            return ""
        return text

    @staticmethod
    def _collapse_repeats(text: str) -> str:
        """같은 단어/구가 3번 이상 연속 반복되면 한 번으로 줄인다."""
        # 단어 1개 반복: "바지에 바지에 바지에" → "바지에"
        text = re.sub(r'(\S+)(?:\s+\1){2,}', r'\1', text)
        # 2~4단어 구 반복: "바지에 바 바지에 바 바지에 바" → "바지에 바"
        text = re.sub(r'((?:\S+\s+){1,3}\S+)(?:\s+\1){2,}', r'\1', text)
        return text.strip()
