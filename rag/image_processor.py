"""
image_processor.py
------------------
역할: 사진 폴더의 이미지들을 Gemini Vision으로 분석해서 JSON으로 저장
사용: python image_processor.py
"""

import os
from dotenv import load_dotenv
import json
import time
from pathlib import Path
from google import genai
from google.genai import types

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────────────────────────
PHOTO_DIR      = "./photos"             # 사진 넣는 폴더
OUTPUT_DIR     = "./data"              # JSON 저장 위치

# 지원하는 이미지 확장자
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
# ─────────────────────────────────────────────────────────────────────────────


def setup_gemini():
    """Gemini API 초기화"""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")

    return genai.Client(api_key=api_key)


def get_image_files(patient_id: str) -> list[Path]:
    """
    환자 사진 폴더에서 이미지 파일 목록 반환
    폴더 구조: photos/P001/사진.jpg
    """
    patient_photo_dir = Path(PHOTO_DIR) / patient_id

    if not patient_photo_dir.exists():
        print(f"  사진 폴더 없음: {patient_photo_dir}")
        print(f"  폴더를 만들고 사진을 넣어주세요: {patient_photo_dir.absolute()}")
        return []

    files = [
        f for f in patient_photo_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXT
    ]

    return sorted(files)


def analyze_image(model, image_path: Path, patient_name: str,
                  caption: str = "") -> dict | None:
    """
    사진 1장을 Gemini Vision으로 분석해서 기억 데이터 반환

    Args:
        model       : Gemini 모델 객체
        image_path  : 이미지 파일 경로
        patient_name: 환자 이름
        caption     : 보호자가 추가한 설명 (없으면 빈 문자열)

    Returns:
        기억 딕셔너리 or None (실패 시)
    """

    # 프롬프트 구성
    caption_part = f"\n[보호자 추가 설명]: {caption}" if caption else ""

    prompt = f"""이 사진은 치매 환자 {patient_name} 어르신의 기억 사진입니다.
사진을 분석해서 아래 JSON 형식으로만 응답해주세요. 다른 말은 하지 마세요.

{caption_part}

{{
  "등장인물": [
    {{"추정관계": "딸로 보임", "특징": "40대 여성, 긴 머리"}}
  ],
  "장소": "가정집 거실로 보임",
  "시간": "명절로 보임 (한복 착용)",
  "상황": "가족이 함께 모여 식사하는 장면",
  "감정": "화목하고 행복한 분위기",
  "content": "사진을 바탕으로 한 2-3문장 기억 서술. 한국어 구어체로.",
  "keywords": ["키워드1", "키워드2", "키워드3"],
  "trigger_phrases": ["할머니가 이 사진 보며 할 법한 말"],
  "importance": "high 또는 medium 또는 low",
  "category": "photo"
}}

규칙:
- 확실하지 않은 정보는 "~로 보임" 표현 사용
- content는 반드시 한국어 구어체로 2-3문장
- 없는 정보는 null로 표시
- JSON만 응답, 다른 텍스트 없이"""

    try:
        # 이미지 파일 읽기
        image_data = image_path.read_bytes()
        ext        = image_path.suffix.lower()
        mime_map   = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",  ".webp": "image/webp",
            ".heic": "image/heic", ".heif": "image/heif"
        }
        mime_type = mime_map.get(ext, "image/jpeg")

        # Gemini Vision 호출 (새 SDK)
        response = model.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_data, mime_type=mime_type),
                prompt
            ]
        )

        # JSON 파싱
        raw = response.text.strip()

        # ```json ... ``` 블록 처리
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)
        return result

    except json.JSONDecodeError as e:
        print(f"    ⚠️  JSON 파싱 실패 ({image_path.name}): {e}")
        return None

    except Exception as e:
        err_msg = str(e)
        import re

        # 429 속도 제한
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            wait = 65
            m = re.search(r"retryDelay.*?(\d+)s", err_msg)
            if m:
                wait = int(m.group(1)) + 5
            print(f"    ⏳ 속도 제한. {wait}초 대기 후 재시도...")
            time.sleep(wait)
            return analyze_image(model, image_path, patient_name, caption)

        # 503 서버 과부하 → 30초 대기 후 재시도
        if "503" in err_msg or "UNAVAILABLE" in err_msg:
            print(f"    ⏳ 서버 과부하. 30초 대기 후 재시도...")
            time.sleep(30)
            return analyze_image(model, image_path, patient_name, caption)

        # 400 키 만료 → 10초 대기 후 재시도
        if "400" in err_msg or "API_KEY_INVALID" in err_msg or "expired" in err_msg:
            print(f"    ⏳ 일시적 오류. 10초 대기 후 재시도...")
            time.sleep(10)
            return analyze_image(model, image_path, patient_name, caption)

        print(f"    ⚠️  분석 실패 ({image_path.name}): {e}")
        return None


def process_photo_folder(patient_id: str, patient_name: str,
                          captions: dict = None) -> list[dict]:
    """
    환자 사진 폴더 전체 처리

    Args:
        patient_id  : 환자 ID (예: "P001")
        patient_name: 환자 이름 (예: "김영순")
        captions    : 파일명 → 보호자 설명 딕셔너리
                      예: {"family_2020.jpg": "2020년 추석에 딸 수진이랑 찍은 사진"}

    Returns:
        photo_memories 리스트
    """
    if captions is None:
        captions = {}

    print(f"\n[사진 처리 시작] 환자: {patient_name} ({patient_id})")

    model  = setup_gemini()
    images = get_image_files(patient_id)

    if not images:
        return []

    print(f"  사진 {len(images)}장 발견\n")

    photo_memories = []

    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] 분석 중: {img_path.name}")

        caption = captions.get(img_path.name, "")
        result  = analyze_image(model, img_path, patient_name, caption)

        if result:
            # 표준 형식으로 변환
            memory = {
                "id"         : f"photo_{patient_id}_{i:03d}",
                "source"     : img_path.name,
                "category"   : "photo",
                "importance" : result.get("importance", "medium"),
                "vision_analysis": {
                    "등장인물": result.get("등장인물", []),
                    "장소"    : result.get("장소", ""),
                    "시간"    : result.get("시간", ""),
                    "상황"    : result.get("상황", ""),
                    "감정"    : result.get("감정", "")
                },
                "caregiver_caption": caption,
                "content"          : result.get("content", ""),
                "keywords"         : result.get("keywords", []),
                "trigger_phrases"  : result.get("trigger_phrases", [])
            }
            photo_memories.append(memory)
            print(f"         ✅ {result.get('content', '')[:50]}...")
        else:
            print(f"         ❌ 분석 실패, 건너뜀")

        # 요청 간격 10초 (서버 과부하 방지)
        if i < len(images):
            time.sleep(10)

    print(f"\n  완료: {len(photo_memories)}/{len(images)}장 성공")
    return photo_memories


def save_photo_memories(patient_id: str, photo_memories: list[dict]):
    """
    분석된 사진 기억을 JSON 파일로 저장
    기존 JSON 있으면 photo_memories 부분만 업데이트
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, f"patient_{patient_id}.json")

    # 기존 JSON 있으면 로드, 없으면 기본 구조 생성
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {
            "patient_info"  : {"id": patient_id},
            "memories"      : [],
            "photo_memories": []
        }

    # photo_memories 교체
    data["photo_memories"] = photo_memories

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 저장 완료: {file_path}")
    print(f"   사진 기억 {len(photo_memories)}개 저장됨")


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── 여기만 수정하세요 ──────────────────────────────────────────────────────
    PATIENT_ID   = "P001"
    PATIENT_NAME = "김영순"

    # 보호자가 각 사진에 달아준 설명 (파일명: 설명)
    # 없는 사진은 Gemini가 사진만 보고 분석
    CAPTIONS = {
        "family_2020.jpg"  : "2020년 추석에 딸 수진이랑 찍은 사진",
        "wedding_1975.jpg" : "1975년 통영 성당에서 결혼식 날 사진",
        "grandchild.jpg"   : "손자 민준이 돌잔치 때 찍은 사진",
    }
    # ──────────────────────────────────────────────────────────────────────────

    photo_memories = process_photo_folder(PATIENT_ID, PATIENT_NAME, CAPTIONS)

    if photo_memories:
        save_photo_memories(PATIENT_ID, photo_memories)
