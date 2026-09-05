"""
audio_out.py
------------
mp3 파일을 스피커로 재생. Pi(Linux)에서는 mpg123, Windows에서는 playsound.

Pi 설치: sudo apt install mpg123

스피커 장치 명시:
- ALSA default 가 마이크 카드를 가리키는 등 default 로 재생이 안 되는 경우
  SPEAKER_ALSA_DEVICE 에 USB 스피커 ALSA 이름을 지정.
- 예: "plughw:1,0"  (aplay -l 에서 본 card 번호 사용. plughw 가 자동 리샘플링)
- None 이면 시스템 default 로 재생.
"""

import platform
import shutil
import queue
import threading
import subprocess
import time
import requests
from pathlib import Path

# 🔊 USB 스피커 ALSA 디바이스. None 이면 system default.
SPEAKER_ALSA_DEVICE = "plughw:CARD=UACDemoV10"

# 🔊 재생 볼륨(소프트웨어 증폭). 1.0=원음, 2.0=2배, 0.5=절반.
# 하드웨어 볼륨(amixer)을 먼저 올리고도 작으면 이 값을 키우세요(너무 크면 찢어짐).
SPEAKER_VOLUME = 1.0


def play_mp3(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    system = platform.system()

    if system == "Linux":
        if shutil.which("mpg123"):
            cmd = ["mpg123", "-q"]
            if SPEAKER_ALSA_DEVICE:
                cmd.extend(["-a", SPEAKER_ALSA_DEVICE])
            if SPEAKER_VOLUME != 1.0:                       # -f: 32768=원음
                cmd.extend(["-f", str(int(32768 * SPEAKER_VOLUME))])
            cmd.append(str(p))
            subprocess.run(cmd, check=True)
            return
        if shutil.which("ffplay"):
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
            if SPEAKER_VOLUME != 1.0:
                cmd.extend(["-af", f"volume={SPEAKER_VOLUME}"])
            cmd.append(str(p))
            subprocess.run(cmd, check=True)
            return
        raise RuntimeError(
            "Linux: mpg123가 설치되어 있지 않습니다. `sudo apt install mpg123`"
        )

    if system == "Darwin":
        subprocess.run(["afplay", str(p)], check=True)
        return

    # Windows: playsound (pip install playsound==1.2.2 권장)
    try:
        from playsound import playsound
        playsound(str(p.resolve()))
    except ImportError as e:
        raise RuntimeError(
            "Windows: `pip install playsound==1.2.2` 가 필요합니다."
        ) from e


def play_wav(path: str) -> None:
    """wav 파일을 스피커로 재생. Pi(Linux)에서는 aplay(ALSA), mac에서는 afplay.

    CosyVoice2 TTS 서버는 wav를 반환하므로, mp3용 mpg123이 아니라 wav용 aplay를 쓴다.
    SPEAKER_ALSA_DEVICE(예: "plughw:1,0")로 USB 스피커를 직접 지정.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    system = platform.system()

    if system == "Linux":
        if shutil.which("aplay"):
            cmd = ["aplay", "-q"]
            if SPEAKER_ALSA_DEVICE:
                cmd.extend(["-D", SPEAKER_ALSA_DEVICE])
            cmd.append(str(p))
            subprocess.run(cmd, check=True)
            return
        if shutil.which("ffplay"):
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
            if SPEAKER_VOLUME != 1.0:
                cmd.extend(["-af", f"volume={SPEAKER_VOLUME}"])
            cmd.append(str(p))
            subprocess.run(cmd, check=True)
            return
        raise RuntimeError(
            "Linux: aplay가 없습니다. `sudo apt install alsa-utils`"
        )

    if system == "Darwin":
        subprocess.run(["afplay", str(p)], check=True)
        return

    # Windows: playsound (pip install playsound==1.2.2 권장)
    try:
        from playsound import playsound
        playsound(str(p.resolve()))
    except ImportError as e:
        raise RuntimeError(
            "Windows: `pip install playsound==1.2.2` 가 필요합니다."
        ) from e


def synthesize_and_play_stream(text, url, key, sample_rate=24000, buffer_bytes=96000,
                               on_start=None, spk_id=None):
    """PCM을 백그라운드 스레드로 미리 받아 쌓아두고 재생. 네트워크 흔들림을 흡수한다.

    spk_id: 어떤 화자(목소리)로 합성할지. None 이면 서버 기본 목소리(caregiver).
            기본 목소리로 등록한 가족 음성의 speaker_id 를 넘기면 그 목소리로 말한다.
    """
    t0 = time.perf_counter()
    q = queue.Queue()
    box = {"ttfb": None, "dl_done": None, "err": None, "bytes": 0}

    def downloader():
        try:
            with requests.post(url, headers={"x-api-key": key},
                               json={"text": text, "spk_id": spk_id},
                               stream=True, timeout=60) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=4096):
                    if box["ttfb"] is None:
                        box["ttfb"] = time.perf_counter()
                    box["bytes"] += len(chunk)
                    q.put(chunk)
        except Exception as e:
            box["err"] = e
        finally:
            box["dl_done"] = time.perf_counter()
            q.put(None)

    threading.Thread(target=downloader, daemon=True).start()

    proc = subprocess.Popen(
        ["aplay", "-q", "-f", "S16_LE", "-r", str(sample_rate), "-c", "1",
         "-D", SPEAKER_ALSA_DEVICE, "-"],
        stdin=subprocess.PIPE
    )

    buf = b""
    started = False
    t_first = None
    while True:
        chunk = q.get()
        if chunk is None:
            break
        if not started:
            buf += chunk
            if len(buf) >= buffer_bytes:
                proc.stdin.write(buf)
                buf = b""
                started = True
                t_first = time.perf_counter()
                if on_start:
                    on_start()
        else:
            proc.stdin.write(chunk)

    if not started and buf:
        proc.stdin.write(buf)
        t_first = time.perf_counter()
        if on_start:
            on_start()

    proc.stdin.close()
    proc.wait()
    t_end = time.perf_counter()

    if box["err"]:
        raise box["err"]

    audio_sec = box["bytes"] / (sample_rate * 2)
    print(
        f"   ⏱  TTS  서버첫청크 {(box['ttfb'] - t0):.2f}s"
        f" | 첫소리 {(t_first - t0):.2f}s"
        f" | 다운로드완료 {(box['dl_done'] - t0):.2f}s"
        f" | 재생완료 {(t_end - t0):.2f}s"
        f" | 오디오길이 {audio_sec:.2f}s"
    )
    return {
        "first_sound": t_first - t0,
        "download_done": box["dl_done"] - t0,
        "total": t_end - t0,
        "audio_sec": audio_sec,
    }


if __name__ == "__main__":
    import sys
    play_mp3(sys.argv[1] if len(sys.argv) > 1 else "response.mp3")
