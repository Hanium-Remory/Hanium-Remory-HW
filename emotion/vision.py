"""
비전 파이프라인: 카메라 → 얼굴 검출(YuNet) → crop → 감정 분류(HSEmotion).

- 얼굴 검출: cv2.FaceDetectorYN (YuNet). 가볍고 각도/측면에 비교적 강함.
- 감정 분류: HSEmotion(EmotiEffLib) — AffectNet 학습 EfficientNet-B0. FER+보다 정확.
  ※ HSEmotion은 RGB 컬러 입력을 받음(내부에서 224로 리사이즈 + ImageNet 정규화).
원본 이미지는 저장하지 않고 메모리에서만 처리합니다.
"""
import cv2
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

import config


class Camera:
    """picamera2 래퍼. 한 장씩 빠르게 캡처."""

    def __init__(self, size=config.CAM_SIZE):
        from picamera2 import Picamera2  # 파이 전용. import는 여기서.
        self._picam = Picamera2()
        cfg = self._picam.create_preview_configuration(
            main={"size": size, "format": "RGB888"}
        )
        self._picam.configure(cfg)
        self._picam.start()

    def capture(self):
        """현재 프레임을 BGR ndarray로 반환."""
        # picamera2의 "RGB888"은 실제로 BGR 순서로 배열을 반환 → OpenCV에 그대로 사용.
        return self._picam.capture_array()

    def close(self):
        try:
            self._picam.stop()
        except Exception:
            pass


class EmotionAnalyzer:
    """YuNet으로 얼굴을 찾고 HSEmotion으로 감정을 분류."""

    def __init__(self):
        self._detector = cv2.FaceDetectorYN.create(
            config.YUNET_PATH, "", (320, 320),
            config.FACE_SCORE_THRESHOLD, config.FACE_NMS_THRESHOLD, 5000,
        )
        # 감정 분류기 (AffectNet 학습). 모델은 ~/.hsemotion/ 에서 로드.
        self._fer = HSEmotionRecognizer(model_name=config.HSEMOTION_MODEL)

    def _detect_largest_face(self, frame_bgr):
        """가장 큰 얼굴 1개의 (x, y, w, h, score). 없으면 None."""
        h, w = frame_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, fw, fh = faces[0][:4]
        return int(x), int(y), int(fw), int(fh), float(faces[0][14])

    def _classify(self, face_bgr):
        """crop된 얼굴(BGR) → (label, confidence). HSEmotion은 RGB 입력."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        label, scores = self._fer.predict_emotions(face_rgb, logits=False)
        return label, float(np.max(scores))

    def analyze_frame(self, frame_bgr):
        """프레임 1장 → (label, confidence) 또는 None(얼굴 없음)."""
        det = self._detect_largest_face(frame_bgr)
        if det is None:
            return None
        x, y, fw, fh, _ = det
        # 정사각형으로 확장 + 여백 → 비율 왜곡 없이 crop.
        cx, cy = x + fw / 2.0, y + fh / 2.0
        half = int(max(fw, fh) * 0.65)
        H, W = frame_bgr.shape[:2]
        x1, y1 = max(0, int(cx - half)), max(0, int(cy - half))
        x2, y2 = min(W, int(cx + half)), min(H, int(cy + half))
        face = frame_bgr[y1:y2, x1:x2]
        if face.size == 0:
            return None
        return self._classify(face)

    def best_emotion(self, frames):
        """
        발화 중 캡처한 여러 프레임 → 대표 감정 1개 (신뢰도 가중 다수결).
        반환: (label, label_ko, confidence) 또는 None.
        """
        votes = {}
        for f in frames:
            res = self.analyze_frame(f)
            if res is None:
                continue
            label, conf = res
            votes[label] = votes.get(label, 0.0) + conf
        if not votes:
            return None
        label = max(votes, key=votes.get)
        total = sum(votes.values())
        conf = votes[label] / total if total else 0.0
        return label, config.EMOTION_KO.get(label, label), conf
