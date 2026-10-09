"""
설정 모음. 환경/현장에 맞게 여기 값만 바꾸면 됩니다.
"""
import os

# ── 모델 경로 ─────────────────────────────────────────────
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
YUNET_PATH   = os.path.join(MODELS_DIR, "yunet.onnx")     # 얼굴 검출
FERPLUS_PATH = os.path.join(MODELS_DIR, "ferplus.onnx")   # (구) FER+ — 미사용
# 감정 분류: HSEmotion 모델명. 첫 실행 시 ~/.hsemotion/ 에 자동 다운로드(또는 미리 배치).
HSEMOTION_MODEL = "enet_b0_8_best_afew"

# ── 카메라 ────────────────────────────────────────────────
CAM_SIZE     = (1280, 720)   # 캡처 해상도
# 발화 1회 동안 얼굴 프레임 몇 장을 뜰지 (다수결용). 캡처 자체는 가벼움.
FRAMES_PER_UTTERANCE = 3

# ── 얼굴 검출(YuNet) ──────────────────────────────────────
FACE_SCORE_THRESHOLD = 0.5
FACE_NMS_THRESHOLD   = 0.3

# ── 감정 라벨 ─────────────────────────────────────────────
# [A] HSEmotion 라벨. [B] 모델이 있으면 그쪽 라벨이 우선.

# [A] 지금 — HSEmotion (AffectNet 8감정). 데모용, FER+보다 정확.
EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear",
                  "Happiness", "Neutral", "Sadness", "Surprise"]
EMOTION_KO = {"Anger": "분노", "Contempt": "못마땅함", "Disgust": "불쾌", "Fear": "두려움",
              "Happiness": "기쁨", "Neutral": "중립", "Sadness": "슬픔", "Surprise": "놀람"}

# [B] 직접 학습한 5감정 모델 (emotion/train/). 아래 파일이 있으면 vision.py 가
#     [A] 대신 자동으로 이 모델을 쓴다. 라벨은 ONNX 메타데이터에 같이 들어 있고,
#     아래 목록은 메타데이터가 없을 때만 쓰는 예비값. 한글 라벨 그대로 백엔드로 간다.
CUSTOM_EMOTION_PATH   = os.path.join(MODELS_DIR, "emotion_ko5.onnx")
CUSTOM_EMOTION_LABELS = ["기쁨", "슬픔", "분노", "불안", "중립"]
