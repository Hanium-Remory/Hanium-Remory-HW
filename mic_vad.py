"""
mic_vad.py
----------
Pi 마이크에서 라이브로 녹음하면서 webrtcvad로 발화 시작/종료를 감지.
stt4vad.py 의 VAD 알고리즘을 스트리밍 버전으로 이식.

사용:
    from mic_vad import LiveRecorder
    rec = LiveRecorder()
    wav_path = rec.record_until_silence("input.wav")
"""

import wave
from collections import deque

import numpy as np
import pyaudio
import webrtcvad


class LiveRecorder:
    def __init__(
        self,
        rate: int = 16000,
        chunk_duration_ms: int = 30,
        vad_aggressiveness: int = 2,
        silence_end_ms: int = 1000,
        max_record_seconds: int = 30,
        min_energy_threshold: float = 600.0,
        input_device_index: int | None = None,
    ):
        self.rate = rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(rate * chunk_duration_ms / 1000)  # samples per frame
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        # 발화 종료 판단: 말끝 뒤로 이만큼 '연속 침묵'이 이어지면 종료.
        self.silence_end_ms = silence_end_ms
        self.max_record_seconds = max_record_seconds
        self.min_energy_threshold = min_energy_threshold
        self.input_device_index = input_device_index

    def record_until_silence(
        self,
        output_path: str = "input.wav",
        on_speech_start=None,
        on_speech_end=None,
        max_wait_seconds: float | None = None,
    ) -> str | None:
        """
        마이크에서 듣다가 발화 시작 감지 → 말끝 뒤 silence_end_ms 만큼 침묵 → wav 저장.
        반환: 저장된 wav 경로. 발화 미감지 시 None.

        on_speech_start() : 발화 시작이 감지된 '그 순간' 1회 호출 (감정 샘플링 시작용).
        on_speech_end()   : 발화 종료/녹음 종료 시 1회 호출 (감정 샘플링 종료용).
            두 콜백은 녹음 루프와 동시에 돌아가는 다른 작업(감정 인식)을 걸어두는 용도.
        """
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.input_device_index,
        )

        start_win: deque = deque(maxlen=10)   # 발화 시작 판단용(최근 0.3초)
        pre_buffer: deque = deque(maxlen=10)  # 말 앞부분 잘림 방지

        frames: list[bytes] = []
        recording_started = False
        total_recorded_frames = 0
        max_frame_count = int(self.max_record_seconds * 1000 / self.chunk_duration_ms)

        waited_frames = 0
        max_wait_frames = (
            int(max_wait_seconds * 1000 / self.chunk_duration_ms)
            if max_wait_seconds is not None
            else None
        )

        # 말끝 침묵 카운터. 연속 침묵 프레임이 이만큼 쌓이면 종료.
        silence_end_frames = int(self.silence_end_ms / self.chunk_duration_ms)
        silence_run = 0   # 연속 침묵 프레임 수
        speech_run = 0    # 연속 음성 프레임 수 (단발 노이즈로 침묵 카운터가 리셋되는 것 방지)

        print("🎙️  마이크 대기 중... (말하면 녹음 시작)")

        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16)

                # RMS 에너지
                audio_float = chunk.astype(np.float32)
                energy = float(np.sqrt(np.mean(audio_float ** 2)))
                if np.isnan(energy):
                    energy = 0.0

                is_speech = self.vad.is_speech(data, self.rate)
                valid_speech = is_speech and energy > self.min_energy_threshold

                if not recording_started:
                    waited_frames += 1

                    if (
                        max_wait_frames is not None
                        and waited_frames >= max_wait_frames
                    ):
                        print("💤 대화 대기시간 초과")
                        break

                    pre_buffer.append(data)
                    start_win.append(1 if valid_speech else 0)
                    if len(start_win) == start_win.maxlen and sum(start_win) > 5:
                        print("▶️  말 시작 감지")
                        recording_started = True
                        frames.extend(pre_buffer)
                        pre_buffer.clear()
                        silence_run = 0
                        speech_run = 0
                        if on_speech_start is not None:
                            try:
                                on_speech_start()
                            except Exception as e:
                                print(f"⚠️  on_speech_start 콜백 오류: {e}")
                else:
                    frames.append(data)
                    total_recorded_frames += 1

                    # 말끝 침묵 누적. 단, '지속된' 음성(2프레임 이상)일 때만 침묵 카운터 리셋
                    # → 침묵 구간에 끼는 단발성 VAD 오탐 때문에 종료가 안 되던 문제 해결.
                    if valid_speech:
                        speech_run += 1
                        if speech_run >= 2:
                            silence_run = 0
                    else:
                        speech_run = 0
                        silence_run += 1

                    if silence_run >= silence_end_frames:
                        print("⏹️  말 종료 감지")
                        break

                    if total_recorded_frames >= max_frame_count:
                        print("⏹️  최대 녹음 길이 도달")
                        break
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            # 발화가 한 번이라도 시작됐다면 종료 콜백 보장(정상 종료/최대길이/예외 모두)
            if recording_started and on_speech_end is not None:
                try:
                    on_speech_end()
                except Exception as e:
                    print(f"⚠️  on_speech_end 콜백 오류: {e}")

        if not frames:
            print("⚠️  감지된 음성이 없습니다.")
            return None

        self._save_wav(output_path, frames)
        return output_path

    def _save_wav(self, output_path: str, frames: list[bytes]) -> None:
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(frames))
        print(f"💾  녹음 저장: {output_path}")


if __name__ == "__main__":
    rec = LiveRecorder()
    path = rec.record_until_silence("mic_test.wav")
    print(f"결과: {path}")
