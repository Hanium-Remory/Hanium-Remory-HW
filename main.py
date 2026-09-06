"""
main.py
-------
모리 인형 통합 파이프라인.

흐름:
    [마이크 라이브 VAD 녹음] → [Whisper STT] → [RAG 컨텍스트]
    → [Groq LLM (reply + 로봇표정)] → [CosyVoice2 TTS (2080 서버)] → [스피커 재생]
                                   └─▶ [LCD에 로봇 표정 표시]

    + [약 복용 시간 백그라운드 알림]
    + [가족 채팅 전달: 글=TTS, 사진=화면]
    + [대화중/연결중 상태 보고]

실행:
    python main.py
"""

import os
import queue
import requests
import threading
import datetime
from contextlib import contextmanager
from dotenv import load_dotenv
import sys
import time
from pathlib import Path
from typing import TypedDict, Literal
import subprocess

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# ⚙️ 설정값
# ──────────────────────────────────────────────────────────────────────────────
PATIENT_ID    = "P001"                       # RAG/chroma_db/ 안에 폴더로 존재해야 함
WHISPER_MODEL = "small"                      # (참고) 실제 STT 모델은 Hailo .hef 로 정해짐
# 목록에서 확인된 정확한 이름으로. (계정에 따라 openai/ 접두어가 붙을 수 있음)
GROQ_MODEL    = "openai/gpt-oss-120b"
# CosyVoice2 TTS 서버(2080). URL/키는 .env로 주입 가능(코드에 고정 X).
TTS_API_URL   = os.getenv("TTS_API_URL", "http://100.101.194.59:8000/tts")
TTS_STREAM_API_URL = os.getenv("TTS_STREAM_API_URL", "http://100.101.194.59:8001/tts_stream")
TTS_API_KEY   = os.getenv("TTS_API_KEY", "morri1234")
RAG_TOP_K     = 3

# ── 백엔드(ReMory 서버) 연결 ──────────────────────────────────
REMORY_API    = os.getenv("REMORY_API", "https://remory-passkey-hanium.onrender.com")
DEVICE_ID     = os.getenv("DEVICE_ID", "1")
DEVICE_TOKEN  = os.getenv("DEVICE_TOKEN", "Fy5JbTV4OFZ9eq6cwoF7cD8IMe1_0aG6le0y0lF1fAM")
# 약 시간을 몇 초마다 확인할지.
MEDICATION_CHECK_INTERVAL_SEC = 30
# 약 알림 후 몇 초 뒤에 "드셨어요?" 하고 복용을 다시 챙겨줄지(기본 15분).
MED_CONFIRM_DELAY_SEC = 15 * 60
# 하트비트('살아있음' 신호)를 몇 초마다 보낼지. 서버는 600초 넘게 끊기면 '연결 끊김'으로 본다.
HEARTBEAT_INTERVAL_SEC = 300
# 서버에서 볼륨·방해금지를 몇 초마다 읽어올지
SETTINGS_SYNC_INTERVAL_SEC = 60
# 가족 채팅을 몇 초마다 확인할지
CHAT_CHECK_INTERVAL_SEC = 15
# 가족 사진을 화면에 몇 초 동안 보여줄지
PHOTO_DISPLAY_SEC = 60
# 어르신 추억(RAG)을 백엔드에서 몇 초마다 동기화할지
RAG_SYNC_INTERVAL_SEC = 300
# 스피커 볼륨 조절용 ALSA 컨트롤 이름 (amixer scontrols 로 확인)
AUDIO_CONTROL = os.getenv("AUDIO_CONTROL", "PCM")
# 서버에서 읽어온 최신 설정 캐시
# spk_id: 기본 목소리의 화자 id(등록한 가족 음성). None 이면 서버 기본 목소리로 말함.
device_settings = {"dnd": None, "volume": None, "spk_id": None}
# 대화 TTS·약 알림·채팅 음성이 동시에 스피커로 나가지 않도록 하는 잠금.
speaker_lock = threading.Lock()
# 사용자가 말하는 중(녹음)인지. 채팅 알림을 이 동안엔 미룬다(그 턴 끝나면 전달).
recording = threading.Event()

# 마이크 디바이스 인덱스. None이면 시스템 default 입력 사용(권장).
MIC_DEVICE_INDEX = None

WAKEWORD_DEVICE_INDEX = None
WAKEWORD_THRESHOLD = 0.65

CONVERSATION_IDLE_TIMEOUT_SEC = 15

# 발화 종료 판단: 말끝 뒤로 이만큼 '연속 침묵'이 이어지면 끝으로 봄(ms).
MIC_SILENCE_END_MS = 2000

# ⏱️ 타이밍 리포트에 표시할 단계 순서.
TIMING_ORDER = ["녹음+VAD", "STT", "RAG", "LLM", "TTS 합성", "TTS 재생"]
# ──────────────────────────────────────────────────────────────────────────────


# 하이픈 들어간 폴더는 직접 import 불가 → sys.path 주입
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "stt"))
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "emotion"))
sys.path.insert(0, str(ROOT / "face"))

from stt4vad_hat import STTHandler              # noqa: E402
from retriever import build_context_prompt  # noqa: E402

from groq import Groq                        # noqa: E402
from mic_vad import LiveRecorder            # noqa: E402
from audio_out import play_wav, synthesize_and_play_stream   # noqa: E402
from wakeword import MoriyaWakeWordDetector
try:
    from emotion_service import EmotionService   # noqa: E402
except Exception as _e:                          # pragma: no cover
    EmotionService = None
    print(f"⚠️  감정 모듈 import 실패 → 감정 없이 동작: {_e}")

try:
    from face_display_2 import FaceDisplay2 as FaceDisplay   # noqa: E402
except Exception as _e:                          # pragma: no cover
    FaceDisplay = None
    print(f"⚠️  표정 표시 모듈 import 실패 → 표정 없이 동작: {_e}")


# RAG retriever.py 안에서 "./chroma_db" 상대경로를 쓰므로 CWD를 RAG 폴더로 이동
os.chdir(ROOT / "rag")


SYSTEM_PROMPT = """당신은 치매 어르신의 말동무인 AI 인형 모리예요.
목표는 '안심시키기'가 아니라 '편하고 즐거운 일상 대화를 계속 이어가기'입니다.

먼저 어르신의 말이 어떤 종류인지 스스로 판단해서 다르게 응답하세요.

[A] 무언가를 기억하지 못해 묻거나 떠올리려 할 때 (이름·날짜·사람·장소·옛일 등. 예: "딸 이름이 뭐였더라", "내가 어디 살았지"):
- 먼저 아래 [관련 기억]에서 답을 찾으세요. 답이 있으면 "수진이요, 큰따님 이름이 수진이잖아요"처럼 그 내용을 다정하게 '직접 알려주세요'. 이때 어르신에게 절대 되묻지 마세요.
- 알려준 뒤에는 그 기억과 이어지는 따뜻한 이야기나 질문으로 대화를 계속 이어가세요. (예: "수진이가 지난번에 왔을 때 좋으셨죠?")
- [관련 기억]에 그 답이 정말 없을 때만 "음, 저도 가물가물한데 같이 떠올려볼까요?"처럼 부드럽게 넘기세요. 답을 지어내지는 마세요.

[B] 그 외 평범한 일상 대화일 때:
- 어르신 말에 먼저 가볍게 공감·맞장구치고, 이어지는 질문을 하나 던져 대화를 계속 이어가세요. (예: 식사, 날씨, 좋아하는 음식·노래·꽃, 옛날 이야기, 오늘 기분 등)
- [관련 기억]이 있으면 그 내용을 화제로 자연스럽게 활용하세요.

공통:
- 매 응답은 되도록 끝에 질문 하나로 마무리해서 어르신이 계속 말하게 하세요.
- 자연스럽고 다정한 반말 섞인 존댓말로, 친구처럼 편하게 이야기하세요.

지켜야 할 것:
- 오직 한국어로만, 순수 한글로만 대답하세요. 한자(漢字)나 중국어 문자, 일본어 문자를 절대 쓰지 마세요. 예: '名字'(X) → '이름'(O), '딸們'(X) → '딸'(O).
- 2~3문장 이내로 짧게 말하세요(공감 한 마디 + 질문 한 개 정도).
- 사실을 지어내지 마세요. 단, [관련 기억]에 있는 내용은 자신 있게 직접 알려주세요.
- 추상적이거나 철학적인 말은 쓰지 마세요.
- [환자 표정] 정보가 있으면 분위기에 맞춰 말투를 조절하되, 표정을 직접 언급하지는 마세요.

안심시키기(평소엔 쓰지 말 것):
- 어르신이 불안해하거나 무서워할 때만 "괜찮아요, 제가 여기 있어요"처럼 곁에 있음을 짧게 강조하세요.
- 집에 가고 싶어 하거나 가족을 찾을 때만 "조금만 있으면 가족이 올 거예요"처럼 부드럽게 안심시키고, 곧바로 일상 화제로 다시 돌아가세요.
- 어르신이 불안해하지 않는 평범한 대화에서는 위 안심·가족 멘트를 절대 먼저 꺼내지 마세요.

[로봇 표정 선택]
응답을 할 때마다 모리(당신)가 지어야 할 표정을 아래 6가지 중에서 하나 고르세요.
이것은 어르신의 감정이 아니라 *모리가 지을 표정*입니다. 어르신이 슬프면 모리는 함께
슬퍼하거나 위로하는 표정을 짓는 식입니다.
- "기쁨"  : 어르신이 즐거운 이야기를 하거나 좋은 소식을 전할 때, 같이 기뻐하는 표정.
- "슬픔"  : 어르신이 슬프거나 외로워할 때 함께 슬퍼하는 표정(공감).
- "위로"  : 어르신이 불안·두려움을 느끼거나 집·가족을 그리워할 때 다정히 위로하는 표정.
- "경청"  : 평범한 일상 대화, 어르신 이야기를 가만히 들어드리는 기본 표정.
- "놀람"  : 흥미로운 소식·놀라운 일을 들어 반응할 때.
- "평온"  : 위 어디에도 해당하지 않는 잔잔한 기본 표정."""


ROBOT_EXPRESSIONS = ["기쁨", "슬픔", "위로", "경청", "놀람", "평온"]


class MoriReply(TypedDict):
    reply: str
    expression: Literal["기쁨", "슬픔", "위로", "경청", "놀람", "평온"]


def warmup() -> None:
    """RAG DB 사전 로드 (첫 검색이 ~12초 걸리는 걸 미리 처리)."""
    print("🔥 RAG 워밍업 중...")
    _ = build_context_prompt(PATIENT_ID, "안녕", top_k=1)
    print("✅ 워밍업 완료")


def chat_with_memory(client: Groq, user_text: str, context: str,
                     emotion: dict | None = None) -> dict:
    """RAG 컨텍스트 + (있으면) 감정 상태를 system prompt에 합쳐서 Groq 호출."""
    full_system = f"{SYSTEM_PROMPT}\n\n{context}"
    if emotion and emotion.get("label") not in (None, "unknown"):
        full_system += (
            f"\n\n[환자 표정] 지금 환자의 표정은 '{emotion['label_ko']}'으로 보입니다"
            f"(신뢰도 {emotion['confidence']:.0%}, 표본 {emotion.get('n', 0)}장)."
        )
    full_system += (
        "\n\n[출력 형식] 반드시 아래 JSON 형식으로만 답하세요. 다른 설명 없이 JSON만 출력합니다.\n"
        '{"reply": "어르신께 할 말", "expression": "기쁨|슬픔|위로|경청|놀람|평온 중 하나"}'
    )
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": full_system},
            {"role": "user",   "content": user_text},
        ],
        temperature=0.8,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    import json
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        data = {}

    reply = (data.get("reply") or "").strip()
    expression = data.get("expression") or "평온"
    if expression not in ROBOT_EXPRESSIONS:
        expression = "평온"
    return {"reply": reply, "expression": expression}


def speak(text: str) -> None:
    """CosyVoice2 스트리밍 TTS(8001)로 문장을 말한다. 약알림·채팅 안내 공용.

    기본 목소리로 지정된 화자(device_settings["spk_id"])로 말한다. None 이면 서버 기본.
    """
    synthesize_and_play_stream(text, TTS_STREAM_API_URL, TTS_API_KEY,
                               spk_id=device_settings.get("spk_id"))


def heartbeat_worker() -> None:
    """주기적으로 '인형이 살아있음'을 서버에 알린다."""
    headers = {"X-Device-Token": DEVICE_TOKEN}
    while True:
        try:
            requests.patch(
                f"{REMORY_API}/devices/{DEVICE_ID}/heartbeat",
                headers=headers, timeout=30,
            )
        except Exception as e:
            print(f"⚠️  서버 연결 확인 실패: {e}")
        time.sleep(HEARTBEAT_INTERVAL_SEC)


def report_emotion(emotion: dict) -> None:
    """인형이 읽은 어르신 감정을 백엔드에 저장한다."""
    label = emotion.get("label_ko")
    if not label or label == "unknown":
        return
    try:
        requests.post(
            f"{REMORY_API}/devices/{DEVICE_ID}/emotions",
            json={"emotion": label},
            headers={"X-Device-Token": DEVICE_TOKEN},
            timeout=30,
        )
    except Exception as e:
        print(f"⚠️  감정 기록 실패: {e}")


def report_utterances(user_text: str, reply: str) -> None:
    """방금 주고받은 말 한 턴을 백엔드에 남긴다.

    리포트를 만드는 재료다. 서버가 7일 지난 것은 알아서 지우므로 인형은
    그냥 보내기만 하면 된다. 실패해도 대화는 그대로 이어간다.
    """
    lines = []
    if (user_text or "").strip():
        lines.append({"speaker": "user", "content": user_text.strip()})
    if (reply or "").strip():
        lines.append({"speaker": "mori", "content": reply.strip()})
    if not lines:
        return
    try:
        requests.post(
            f"{REMORY_API}/devices/{DEVICE_ID}/utterances",
            json={"utterances": lines},
            headers={"X-Device-Token": DEVICE_TOKEN},
            timeout=30,
        )
    except Exception as e:
        print(f"⚠️  발화 기록 실패: {e}")


def report_activity(activity_type: str, content: str | None = None) -> None:
    """어르신 일과 한 줄을 남긴다(앱 홈·리포트의 '일과')."""
    try:
        requests.post(
            f"{REMORY_API}/devices/{DEVICE_ID}/activities",
            json={"activityType": activity_type, "content": content},
            headers={"X-Device-Token": DEVICE_TOKEN},
            timeout=30,
        )
    except Exception as e:
        print(f"⚠️  활동 기록 실패: {e}")


# 대화중 보고 대기열. 웨이크워드 직후에 보내는 신호라 녹음 시작을 막으면 안 되고,
# 시작/종료가 뒤집혀 도착하면 앱이 계속 '대화중'으로 남는다. 그래서 한 줄로 세워 보낸다.
_conversation_queue: "queue.Queue[bool]" = queue.Queue()


def send_conversation(active: bool) -> None:
    """대화 시작/종료를 서버에 바로 보낸다(응답을 기다린다)."""
    try:
        requests.patch(
            f"{REMORY_API}/devices/{DEVICE_ID}/conversation",
            json={"active": active},
            headers={"X-Device-Token": DEVICE_TOKEN}, timeout=10,
        )
    except Exception as e:
        print(f"⚠️  대화상태 보고 실패: {e}")


def conversation_worker() -> None:
    """대기열에 쌓인 대화 시작/종료 보고를 들어온 순서대로 서버에 보낸다."""
    while True:
        send_conversation(_conversation_queue.get())


def report_conversation(active: bool) -> None:
    """대화 시작(True)/종료(False)를 서버에 알린다(앱의 '대화중' 표시용).

    네트워크가 느려도 어르신 말을 놓치지 않도록, 대기열에 넣고 바로 돌아온다.
    """
    _conversation_queue.put(active)


def _speak_when_free(
    text: str, face=None, notice_title: str | None = None, notice_sub: str = ""
) -> None:
    """어르신이 말하는 중이면 그 턴이 끝나길 기다렸다가 음성으로 안내한다.

    [notice_title] 을 주면 말하는 동안 화면에 안내를 띄우고, 말이 끝나면
    내려서 모리 얼굴로 돌아온다. 소리를 놓쳤거나 잘 안 들리셔도 무슨
    일인지 읽을 수 있게 하려는 것이다.
    """
    while recording.is_set():
        time.sleep(0.3)
    with speaker_lock:
        if face:
            face.set_expression("평온")
            if notice_title:
                face.show_notice(notice_title, notice_sub)
        try:
            speak(text)
        finally:
            # 말이 끊기든 끝나든 안내가 화면에 남지 않게 한다.
            if face and notice_title:
                face.hide_notice()


def _confirm_medication(med_name: str, face=None) -> None:
    """약 알림 MED_CONFIRM_DELAY_SEC 후, 복용했는지 한 번 더 챙겨 묻는다(대답은 안 받음)."""
    _speak_when_free(
        f"{med_name} 드셨어요? 아직 안 드셨으면 지금 꼭 챙겨 드세요.",
        face,
        notice_title="약 드셨어요?",
        notice_sub=med_name,
    )


def medication_worker(face=None) -> None:
    """백그라운드에서 약 복용 시간을 지켜보다가 때가 되면 음성으로 안내한다."""
    headers = {"X-Device-Token": DEVICE_TOKEN}
    alerted = set()   # (약id, 날짜) — 하루 한 번만
    while True:
        try:
            now = datetime.datetime.now()
            hhmm = now.strftime("%H:%M")
            today = now.date().isoformat()

            r = requests.get(
                f"{REMORY_API}/devices/{DEVICE_ID}/medications",
                headers=headers, timeout=30,
            )
            data = r.json().get("data", {})
            med_check = data.get("medicationCheck", True)   # 15분 뒤 복용 확인을 할지

            for m in data.get("medications", []):
                key = (m["medicationId"], today)
                if m.get("enabled") and m["time"] == hhmm and key not in alerted:
                    alerted.add(key)
                    timing = (m.get("timing") or "").strip()
                    if timing:
                        text = f"{timing}에 {m['name']} 드실 시간이에요. 잊지 말고 꼭 챙겨 드세요."
                    else:
                        text = f"{m['name']} 드실 시간이에요. 잊지 말고 꼭 챙겨 드세요."
                    _speak_when_free(
                        text, face, notice_title="약 드실 시간이에요!", notice_sub=m["name"]
                    )
                    # 앱이 아이콘·문구를 코드로 고르므로 한글이 아니라 코드로 남긴다.
                    report_activity("MEDICATION", m["name"])

                    # 복용 확인이 켜져 있으면 15분 뒤에 한 번 더 챙겨 묻는다.
                    if med_check:
                        threading.Timer(
                            MED_CONFIRM_DELAY_SEC, _confirm_medication,
                            args=(m["name"], face),
                        ).start()
        except Exception as e:
            print(f"⚠️  약 알림 오류: {e}")

        time.sleep(MEDICATION_CHECK_INTERVAL_SEC)


# 사진을 내릴 타이머. 사진이 잇따라 오면 앞 메시지의 타이머가 뒤 사진을
# 일찍 내려버리므로, 새 사진을 띄울 때 앞 타이머를 취소한다.
_photo_timer: threading.Timer | None = None
_photo_timer_lock = threading.Lock()


def _show_photo_for_a_while(face, url: str) -> None:
    """가족 사진을 띄우고 PHOTO_DISPLAY_SEC 뒤에 내려 모리 얼굴로 돌아온다."""
    global _photo_timer
    with _photo_timer_lock:
        if _photo_timer is not None:
            _photo_timer.cancel()
        face.show_photo(url)
        _photo_timer = threading.Timer(PHOTO_DISPLAY_SEC, face.hide_photo)
        _photo_timer.daemon = True
        _photo_timer.start()


def _deliver_chat(m: dict, face=None) -> None:
    """가족 메시지 하나를 인형이 전한다. 글=TTS, 사진=화면(PHOTO_DISPLAY_SEC 초)."""
    content = (m.get("content") or "").strip()
    image_url = m.get("imageUrl")

    # 보낸 사람 호칭 만들기: "딸 박수진" → "딸 박수진에게서". 정보 없으면 "가족".
    relation = (m.get("senderRelation") or "").strip()
    name = (m.get("senderName") or "").strip()
    if name:
        sender = f"{relation} {name}".strip()   # 관계 없으면 이름만
    else:
        sender = "가족"

    # 사용자가 말하는 중이면 그 턴이 끝날 때까지 기다린다(사용자 말을 끊지 않음).
    while recording.is_set():
        time.sleep(0.3)

    # 모리가 말하는 중이면 speaker_lock 이 그 말이 끝날 때까지 기다려 준다.
    with speaker_lock:
        if face:
            face.set_expression("기쁨")
            # 소리를 놓쳐도 누가 무슨 말을 했는지 읽을 수 있게 화면에도 띄운다.
            if content:
                face.show_notice(content, f"{sender}에게서", icon="message")
            elif image_url:
                face.show_notice("사진이 왔어요", f"{sender}에게서", icon="message")
        try:
            if content:
                speak(f"{sender}에게서 메시지가 왔어요. {content}")
            if image_url:
                speak(f"{sender}이 사진을 보냈어요. 화면을 봐 주세요.")
        finally:
            # 말이 끊기든 끝나든 안내가 화면에 남지 않게 한다.
            if face:
                face.hide_notice()
                face.set_expression("평온")

    # 사진은 화면에 잠깐 띄우고, 일정 시간 뒤 자동으로 내린다(음성과 별개로 계속 표시).
    if image_url and face:
        try:
            _show_photo_for_a_while(face, image_url)
        except Exception as e:
            print(f"⚠️  사진 표시 실패: {e}")


def chat_worker(face=None) -> None:
    """가족이 보낸 메시지(글·사진)를 받아 인형이 전한다."""
    headers = {"X-Device-Token": DEVICE_TOKEN}
    while True:
        try:
            r = requests.get(
                f"{REMORY_API}/devices/{DEVICE_ID}/chat/pending",
                headers=headers, timeout=30,
            )
            msgs = (r.json().get("data") or {}).get("messages", [])
            if msgs:
                for m in msgs:
                    _deliver_chat(m, face)
                ids = [m["messageId"] for m in msgs]
                requests.post(
                    f"{REMORY_API}/devices/{DEVICE_ID}/chat/delivered",
                    json={"messageIds": ids}, headers=headers, timeout=30,
                )
        except Exception as e:
            print(f"⚠️  채팅 전달 오류: {e}")
        time.sleep(CHAT_CHECK_INTERVAL_SEC)


def _make_vision_model():
    """사진 비전분석용 Gemini 모델. RAG_USE_VISION=true 이고 키가 있을 때만 만든다.

    키가 무효면 분석 호출이 계속 재시도로 멈추므로, 기본은 꺼둔다(사진 설명으로 대체).
    유효한 Gemini 키가 있으면 .env 에 RAG_USE_VISION=true 로 켜면 된다.
    """
    if os.getenv("RAG_USE_VISION", "false").lower() != "true":
        return None
    if not os.getenv("GEMINI_API_KEY"):
        return None
    try:
        from image_processor import setup_gemini
        return setup_gemini()
    except Exception as e:
        print(f"⚠️  비전분석 모델 초기화 실패 → 사진 설명만 사용: {e}")
        return None


def run_rag_sync(vision_model) -> bool:
    """백엔드 어르신 데이터(사진·가족)로 chroma 를 한 번 갱신한다."""
    try:
        from rag_sync import sync_rag
    except Exception as e:
        print(f"⚠️  rag_sync 모듈 로드 실패 → RAG 동기화 생략: {e}")
        return False
    try:
        if sync_rag(PATIENT_ID, REMORY_API, DEVICE_ID, DEVICE_TOKEN, vision_model):
            print("🧠 RAG 동기화 완료")
            return True
    except Exception as e:
        print(f"⚠️  RAG 동기화 오류: {e}")
    return False


def rag_sync_worker(vision_model):
    """주기적으로 RAG 를 갱신한다(앱에 올린 사진·추억을 인형 대화에 반영)."""
    while True:
        time.sleep(RAG_SYNC_INTERVAL_SEC)
        run_rag_sync(vision_model)


def apply_volume(vol):
    """스피커 볼륨을 vol(%)로 설정 (ALSA amixer)."""
    try:
        subprocess.run(["amixer", "-q", "sset", AUDIO_CONTROL, f"{int(vol)}%"], check=False)
        print(f"🔊 볼륨 {vol}% 적용")
    except Exception as e:
        print(f"⚠️  볼륨 설정 실패: {e}")


def in_dnd(dnd, now=None):
    """지금이 방해 금지 시간대인지."""
    if not dnd or not dnd.get("enabled"):
        return False
    now = now or datetime.datetime.now()
    h, s, e = now.hour, dnd.get("startHour", 23), dnd.get("endHour", 7)
    if s == e:
        return False
    return (s <= h < e) if s < e else (h >= s or h < e)


def settings_worker():
    """서버에서 볼륨·방해금지·기본 목소리(speaker_id)를 주기적으로 읽어 반영한다."""
    headers = {"X-Device-Token": DEVICE_TOKEN}
    while True:
        try:
            r = requests.get(f"{REMORY_API}/devices/{DEVICE_ID}/settings", headers=headers, timeout=30)
            data = r.json().get("data") or {}
            vol = data.get("volume")
            if vol is not None and vol != device_settings["volume"]:
                apply_volume(vol)
                device_settings["volume"] = vol

            # 기본 목소리(defaultVoiceId)의 speaker_id 를 뽑아둔다.
            # 인형이 TTS 할 때 이 화자로 말한다. speaker_id 가 없으면(기본 제공 음성)
            # None → 서버 기본 목소리(caregiver)로 말함.
            default_id = data.get("defaultVoiceId")
            spk = None
            for v in data.get("voices", []):
                if v.get("voiceId") == default_id:
                    spk = v.get("speakerId")
                    break
            if spk != device_settings["spk_id"]:
                device_settings["spk_id"] = spk
                print(f"🎙️  기본 목소리 speaker_id = {spk}")

            r = requests.get(f"{REMORY_API}/devices/{DEVICE_ID}/dnd", headers=headers, timeout=30)
            device_settings["dnd"] = r.json().get("data")
        except Exception as e:
            print(f"⚠️  설정 동기화 오류: {e}")
        time.sleep(SETTINGS_SYNC_INTERVAL_SEC)


@contextmanager
def timed(phase: str, store: dict):
    """`with timed("STT", timings):` 블록의 경과시간(초)을 store[phase]에 기록."""
    t = time.perf_counter()
    try:
        yield
    finally:
        store[phase] = time.perf_counter() - t


def print_timing_report(timings: dict, extra_info: str = "") -> None:
    """턴 끝에 단계별 + 총합 시간을 표/막대로 출력."""
    if not timings:
        return
    header = "┌─ ⏱️  타이밍 (이번 턴)"
    if extra_info:
        header += f"  ({extra_info})"
    print(header)

    seen = set()
    total = 0.0
    for k in TIMING_ORDER:
        if k in timings:
            v = timings[k]
            bar = "█" * max(1, int(v * 4))
            print(f"│  {k:<10}{v:7.2f}s  {bar}")
            total += v
            seen.add(k)
    for k, v in timings.items():
        if k in seen:
            continue
        bar = "█" * max(1, int(v * 4))
        print(f"│  {k:<10}{v:7.2f}s  {bar}")
        total += v

    print(f"│  {'─' * 34}")
    print(f"│  {'총합':<10}{total:7.2f}s")
    print("└" + "─" * 44)


def main() -> None:
    print("=" * 50)
    print("  🐻 모리 통합 파이프라인 시작")
    print(f"  환자 ID: {PATIENT_ID}")
    print("=" * 50)

    stt           = STTHandler(model_size=WHISPER_MODEL)

    wakeword_detector = MoriyaWakeWordDetector(
        moriya_model_path=ROOT / "models" / "moriya_v1.onnx",
        models_dir=ROOT / "models",
        threshold=WAKEWORD_THRESHOLD,
        input_device_index=WAKEWORD_DEVICE_INDEX,
    )

    emotion_svc = None
    if EmotionService is not None:
        try:
            emotion_svc = EmotionService()
            print("📷 감정 인식 활성화")
        except Exception as e:
            print(f"⚠️  감정 인식 비활성화(카메라/모델 초기화 실패): {e}")

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # RAG 초기 동기화(백엔드 데이터로 chroma 생성) → 그 다음 워밍업.
    vision_model = _make_vision_model()
    print("🧠 RAG 초기 동기화 중...")
    run_rag_sync(vision_model)
    try:
        warmup()
    except Exception as e:
        print(f"⚠️  RAG 워밍업 건너뜀(데이터 아직 없음): {e}")

    face = None
    if FaceDisplay is not None:
        try:
            face = FaceDisplay()
            face.set_expression("평온")
            print("🖥️  표정 디스플레이 활성화")
        except Exception as e:
            print(f"⚠️  표정 디스플레이 비활성화: {e}")

    # 💬 대화중 보고 워커 (웨이크워드 직후 보고가 녹음을 막지 않게 따로 보낸다)
    threading.Thread(target=conversation_worker, daemon=True).start()

    # 시작 시엔 '대화중' 아님으로 맞춰둔다(이전 실행이 켜둔 채 죽었을 수 있으므로).
    report_conversation(False)

    # 💊 약 복용 알림 워커
    threading.Thread(target=medication_worker, args=(face,), daemon=True).start()
    print("💊 약 알림 워커 시작")

    # 💓 서버 연결 확인 워커
    threading.Thread(target=heartbeat_worker, daemon=True).start()
    print("💓 연결 확인 시작")

    # ⚙️ 설정 동기화 워커(볼륨·방해금지·기본 목소리)
    threading.Thread(target=settings_worker, daemon=True).start()
    print("⚙️  설정 동기화 워커 시작")

    # 💬 가족 채팅 워커(글=TTS, 사진=화면)
    threading.Thread(target=chat_worker, args=(face,), daemon=True).start()
    print("💬 가족 채팅 워커 시작")

    # 🧠 RAG 동기화 워커(앱에 올린 사진·추억을 인형 대화에 반영)
    threading.Thread(target=rag_sync_worker, args=(vision_model,), daemon=True).start()
    print("🧠 RAG 동기화 워커 시작")

    mic_wav   = str(ROOT / "input.wav")

    consecutive_errors = 0
    conversation_active = False
    # 이번 대화(웨이크워드~종료)에서 몇 번 주고받았는지. 끝날 때 일과로 남긴다.
    turns_this_session = 0

    while True:
        try:
            if not conversation_active:
                if face:
                    face.set_expression("평온")

                wakeword_detector.wait_for_wakeword()
                dnd = device_settings["dnd"]
                if in_dnd(dnd) and not (dnd or {}).get("allowWakeWord", True):
                    print("🔕 방해 금지 시간 — 조용히 대기")
                    continue
                print("🎤 말씀하세요.")
                if face:
                    # 알아들었다는 걸 바로 보여준다. 소리만으로는 어르신도
                    # 옆에서 보는 가족도 웨이크워드가 먹었는지 알 수 없다.
                    face.set_expression("경청")
                conversation_active = True
                turns_this_session = 0
                report_conversation(True)   # 웨이크워드~ = 대화중 시작

            recorder = LiveRecorder(
                input_device_index=MIC_DEVICE_INDEX,
                silence_end_ms=MIC_SILENCE_END_MS,
            )

            emotion_holder: dict = {"result": None}
            timings: dict = {}

            def on_start():
                if emotion_svc:
                    emotion_svc.on_speech_start()

            def on_end():
                if emotion_svc:
                    emotion_holder["result"] = emotion_svc.on_speech_end()

            # ① 마이크 라이브 녹음 (녹음 중엔 recording 플래그 ON → 채팅 알림 미룸)
            recording.set()
            if face:
                face.listen_start()
            try:
                with timed("녹음+VAD", timings):
                    wav_path = recorder.record_until_silence(
                        mic_wav,
                        on_speech_start=(on_start if emotion_svc else None),
                        on_speech_end=(on_end if emotion_svc else None),
                        max_wait_seconds=CONVERSATION_IDLE_TIMEOUT_SEC,
                    )
            finally:
                recording.clear()
                # 다 들었다. 이제 생각하고 말할 차례라 불을 끈다.
                if face:
                    face.listen_stop()

            if wav_path is None:
                print("💤 대화를 종료하고 웨이크워드 대기로 돌아갑니다.")
                conversation_active = False
                report_conversation(False)   # 대화중 종료
                # 한 마디도 못 나눈 헛걸음은 일과에 남기지 않는다.
                if turns_this_session:
                    report_activity(
                        "DAILY_CONVERSATION", f"{turns_this_session}번 주고받았어요"
                    )
                    turns_this_session = 0
                continue

            # ② STT
            with timed("STT", timings):
                user_text, stt_elapsed = stt.transcribe(wav_path)
            print(f"🗣️  이용자: {user_text}   ({stt_elapsed:.2f}s)")
            if not user_text:
                continue

            emotion = emotion_holder["result"]
            if emotion:
                print(f"😊 감정: {emotion['label_ko']} "
                      f"({emotion['confidence']:.0%}, {emotion.get('n', 0)}장)")
                report_emotion(emotion)

            # ③ RAG (데이터가 아직 없어도 대화는 이어가게 감싼다)
            with timed("RAG", timings):
                try:
                    context = build_context_prompt(PATIENT_ID, user_text, top_k=RAG_TOP_K)
                except Exception as e:
                    print(f"⚠️  RAG 검색 실패(데이터 없음?): {e}")
                    context = ""

            # ④ LLM
            with timed("LLM", timings):
                llm_out = chat_with_memory(groq_client, user_text, context, emotion)
            reply, robot_expr = llm_out["reply"], llm_out["expression"]
            print(f"🐻 모리: {reply}    [표정: {robot_expr}]")

            # 말이 나가는 걸 늦추지 않도록 따로 보낸다.
            turns_this_session += 1
            threading.Thread(
                target=report_utterances, args=(user_text, reply), daemon=True
            ).start()

            if face:
                face.set_expression(robot_expr)

            # ⑤⑥ TTS 스트리밍 (speaker_lock 으로 약알림·채팅과 안 겹치게)
            #     기본 목소리로 지정된 화자(device_settings["spk_id"])로 말한다.
            try:
                with speaker_lock:
                    with timed("TTS 스트리밍", timings):
                        synthesize_and_play_stream(
                            reply,
                            TTS_STREAM_API_URL,
                            TTS_API_KEY,
                            spk_id=device_settings.get("spk_id"),
                            on_start=(face.start_speaking if face else None),
                        )
            finally:
                if face:
                    face.stop_speaking()

            if face:
                face.set_expression("평온")

            extra = f"감정 {emotion.get('n', 0)}장 백그라운드" if emotion else ""
            print_timing_report(timings, extra_info=extra)
            print()
            consecutive_errors = 0

        except KeyboardInterrupt:
            print("\n👋 종료")
            break
        except Exception as e:
            consecutive_errors += 1
            print(f"❌ 에러: {e}")
            if consecutive_errors >= 5:
                print("⛔ 같은 에러가 반복됩니다. 마이크 장치(MIC_DEVICE_INDEX) 설정을 확인하세요. 종료합니다.")
                break
            time.sleep(0.5)
            continue

    # 어떤 이유로 멈추든 앱에 '대화중'이 남지 않게 한다.
    # 곧 프로세스가 끝나므로 대기열 대신 직접 보낸다.
    send_conversation(False)
    if turns_this_session:
        report_activity("DAILY_CONVERSATION", f"{turns_this_session}번 주고받았어요")

    if emotion_svc:
        emotion_svc.close()
    if face:
        face.close()


if __name__ == "__main__":
    main()
