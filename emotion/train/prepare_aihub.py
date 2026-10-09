"""
AI Hub '한국인 감정인식을 위한 복합 영상' 원본 → 얼굴 crop 데이터셋.

인형과 똑같이 YuNet 으로 가장 큰 얼굴을 찾고 face_crop.crop_square 로 자른 뒤
256x256 으로 저장한다. 원본이 수천 픽셀이라 매 epoch 마다 검출하면 너무 느리므로 한 번만 한다.

  python prepare_aihub.py --src ~/data/aihub82 --out data/crops

--src 아래에 AI Hub 에서 받은 zip 을 그대로 두면 된다(풀어 둔 사진·JSON 도 읽음, 폴더 구조는 자유).
zip 은 풀지 않고 안에서 바로 읽는다. 다 풀면 200GB 가 넘고, 파일명이 CP949 라
macOS 에서 풀면 한글이 깨져 라벨 JSON 과 이름이 안 맞는다.
  - 경로에 Validation / VALID 가 들어간 사진 → test (모델 고를 때 절대 보지 않는 최종 시험용)
  - 나머지 → train 과 val 로 9:1. 파일명 맨 앞 해시(사람 단위)로 나눠 같은 사람이
    train·val 에 같이 들어가지 않게 한다(섞이면 val 점수가 실제보다 높게 나옴).

라벨은 라벨 JSON 의 작업자(annot_A/B/C …) 다수결을 우선 쓰고, JSON 에 없는 사진은
경로(폴더명/파일명)에 들어 있는 감정 이름으로 정한다. 작업자 의견이 갈린 사진은 버린다.
출력 폴더는 원래 7감정 이름으로 나눠 두고, 5감정으로 합치는 건 train.py 가 한다.

중간에 끊겨도 다시 실행하면 이미 자른 사진은 건너뛴다.
"""
import argparse
import hashlib
import json
import os
import unicodedata
import zipfile
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

from labels import AIHUB_LABELS, MODELS_DIR
import face_crop

IMG_EXTS = {".jpg", ".jpeg", ".png"}
SAVE_SIZE = 256
# 인형 카메라는 1280x720. 원본을 비슷한 크기로 줄인 뒤 검출·crop 해야 얼굴 해상도가 맞는다.
MAX_SIDE = 1280


def nfc(s):
    # macOS 에서 푼 zip 은 한글 경로가 NFD(자모 분리)라 "기쁨" in path 가 실패한다.
    return unicodedata.normalize("NFC", s)


def zip_name(info):
    """zip 안 파일명. UTF-8 플래그가 없으면 Python 이 cp437 로 읽으므로 CP949 로 되돌린다."""
    if info.flag_bits & 0x800:
        return nfc(info.filename)
    try:
        return nfc(info.filename.encode("cp437").decode("cp949"))
    except UnicodeError:
        return nfc(info.filename)


def iter_files(src, exts):
    """(표시 경로, 파일명, 소스) — 소스는 디스크 경로 또는 (zip 경로, zip 안 이름)."""
    for root, _, files in os.walk(src):
        for f in sorted(files):
            path = os.path.join(root, f)
            if f.lower().endswith(".zip"):
                with zipfile.ZipFile(path) as z:
                    for info in z.infolist():
                        name = zip_name(info)
                        if not info.is_dir() and os.path.splitext(name)[1].lower() in exts:
                            yield f"{path}/{name}", os.path.basename(name), (path, info.filename)
            elif os.path.splitext(f)[1].lower() in exts:
                yield path, nfc(f), path


def read_bytes(source, zips=None):
    if isinstance(source, tuple):
        zpath, member = source
        if zips is None:
            with zipfile.ZipFile(zpath) as z:
                return z.read(member)
        if zpath not in zips:
            zips[zpath] = zipfile.ZipFile(zpath)
        return zips[zpath].read(member)
    with open(source, "rb") as fp:
        return fp.read()


def label_in_path(path):
    found = [e for e in AIHUB_LABELS if e in nfc(path)]
    return found[0] if len(found) == 1 else None


def _records(obj):
    """JSON 안에서 파일명을 가진 dict 들을 찾아낸다(리스트/중첩 구조 모두)."""
    if isinstance(obj, list):
        for x in obj:
            yield from _records(x)
    elif isinstance(obj, dict):
        if any("filename" in k.lower() or "file_name" in k.lower() for k in obj):
            yield obj
        else:
            for v in obj.values():
                if isinstance(v, (list, dict)):
                    yield from _records(v)


def _find_emotion(obj):
    """작업자 항목 안에서 감정 문자열을 찾는다. 배포본은 annot_A.faceExp,
    AI Hub 문서는 faceBB_A.boxes.label 로 적혀 있어 둘 다 받는다."""
    if isinstance(obj, dict):
        for k in ("faceExp", "label"):
            v = obj.get(k)
            if isinstance(v, str) and nfc(v) in AIHUB_LABELS:
                return nfc(v)
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            found = _find_emotion(v)
            if found:
                return found
    return None


def _vote(rec, min_agree):
    votes = [e for k, v in rec.items()
             if k.lower().startswith(("annot", "facebb")) and (e := _find_emotion(v))]
    if not votes:
        up = rec.get("faceExp_uploader")
        return nfc(up) if up else None
    label, n = Counter(nfc(v) for v in votes).most_common(1)[0]
    return label if n >= min_agree else None


def load_json_labels(src, min_agree):
    """{파일명: 라벨 or None(의견 갈림)}"""
    labels = {}
    for path, _, source in iter_files(src, {".json"}):
        try:
            data = json.loads(read_bytes(source).decode("utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"⚠️  JSON 읽기 실패, 건너뜀: {path} ({e})")
            continue
        for rec in _records(data):
            key = next(k for k in rec if "filename" in k.lower() or "file_name" in k.lower())
            labels[nfc(os.path.basename(str(rec[key])))] = _vote(rec, min_agree)
    return labels


def split_of(path, val_ratio):
    p = nfc(path).lower()
    if "validation" in p or "valid" in p:
        return "test"
    # AI Hub 파일명은 "<사람 해시>_<성별>_<나이>_<감정>_…" 이라 맨 앞이 같으면 같은 사람.
    person = os.path.basename(p).split("_")[0]
    h = int(hashlib.md5(person.encode()).hexdigest(), 16) % 1000
    return "val" if h < val_ratio * 1000 else "train"


_detector = None
_zips = {}


def _init_worker():
    global _detector
    cv2.setNumThreads(1)
    _detector = face_crop.create_detector(os.path.join(MODELS_DIR, "yunet.onnx"))


def _process(job):
    source, dst_path = job
    if os.path.exists(dst_path):
        return "skip"
    try:
        buf = read_bytes(source, _zips)
    except (OSError, zipfile.BadZipFile):
        return "bad"
    img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "bad"
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    det = face_crop.detect_largest_face(_detector, img)
    if det is None:
        return "noface"
    face = face_crop.crop_square(img, det)
    if face is None:
        return "noface"
    face = cv2.resize(face, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, face, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="AI Hub zip(또는 푼 파일)이 있는 최상위 폴더")
    ap.add_argument("--out", default="data/crops")
    ap.add_argument("--min-agree", type=int, default=2, help="작업자 몇 명이 같아야 쓸지")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="split·감정별 최대 장수(0=전부). 먼저 작게 돌려볼 때")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    if not os.path.exists(os.path.join(MODELS_DIR, "yunet.onnx")):
        raise SystemExit(f"YuNet 모델이 없습니다: {MODELS_DIR}/yunet.onnx")

    print("라벨 JSON 읽는 중...")
    json_labels = load_json_labels(args.src, args.min_agree)
    print(f"  JSON 라벨 {len(json_labels)}개 (의견 갈림 {sum(v is None for v in json_labels.values())}개)")

    jobs, per_bucket, stat = [], Counter(), Counter()
    for path, name, source in iter_files(args.src, IMG_EXTS):
        if name in json_labels:
            label = json_labels[name]
            if label is None:
                stat["작업자 의견 갈림"] += 1
                continue
        else:
            label = label_in_path(path)
        if label not in AIHUB_LABELS:
            stat["라벨 모름"] += 1
            continue
        split = split_of(path, args.val_ratio)
        if args.max_per_class and per_bucket[split, label] >= args.max_per_class:
            continue
        per_bucket[split, label] += 1
        stem = os.path.splitext(name)[0]
        jobs.append((source, os.path.join(args.out, split, label, stem + ".jpg")))

    for k, v in stat.items():
        print(f"  제외 — {k}: {v}")
    print(f"자를 사진 {len(jobs)}장, 프로세스 {args.workers}개")

    results = Counter()
    with Pool(args.workers, initializer=_init_worker) as pool:
        for r in tqdm(pool.imap_unordered(_process, jobs, chunksize=16), total=len(jobs)):
            results[r] += 1
    print("결과:", dict(results))
    if results["noface"]:
        print(f"  얼굴 못 찾음 {results['noface'] / max(1, len(jobs)):.1%} — 인형도 이런 사진에선 감정을 못 읽는다.")

    print("\nsplit / 감정별 장수:")
    for split in ("train", "val", "test"):
        d = os.path.join(args.out, split)
        if os.path.isdir(d):
            counts = {e: len(os.listdir(os.path.join(d, e)))
                      for e in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, e))}
            print(f"  {split}: {counts}")


if __name__ == "__main__":
    main()
