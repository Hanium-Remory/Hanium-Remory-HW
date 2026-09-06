"""
face_display_2.py  (브라우저 얼굴 버전 — 기존 pygame FaceDisplay 대체)
---------------------------------------------------------------
브라우저(mori_face.html)에 WebSocket으로 표정/말하기/사진 명령을 보낸다.
main.py 는 set_expression / listen_start / listen_stop / start_speaking / stop_speaking /
show_notice / hide_notice / show_photo / hide_photo / close 만 쓴다.

필요: pip install websockets
브라우저는 chromium --kiosk 로 mori_face.html 을 띄워두면 자동 연결된다.
"""
import asyncio, json, threading
import websockets


class FaceDisplay2:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self._host, self._port = host, port
        self._clients = set()
        self._cur_expr = "평온"
        self._listening = False
        self._notice = None          # (제목, 부제) — 안내가 떠 있는 동안만
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)   # 서버가 listen 시작할 때까지 대기

    def _run(self):
        asyncio.set_event_loop(self._loop)

        async def handler(ws, *_):   # websockets v10/v11 시그니처 모두 호환
            self._clients.add(ws)
            try:
                # 새로 연결되면 현재 표정을 즉시 동기화
                await ws.send(json.dumps({"cmd": "emotion", "name": self._cur_expr},
                                         ensure_ascii=False))
                if self._listening:
                    await ws.send(json.dumps({"cmd": "listen_start"}))
                if self._notice:
                    await ws.send(json.dumps(
                        {"cmd": "notice", "title": self._notice[0], "sub": self._notice[1]},
                        ensure_ascii=False))
                async for _msg in ws:
                    pass
            finally:
                self._clients.discard(ws)

        async def boot():
            await websockets.serve(handler, self._host, self._port)
            self._ready.set()

        self._loop.run_until_complete(boot())
        self._loop.run_forever()

    def _send(self, msg: dict):
        data = json.dumps(msg, ensure_ascii=False)

        async def _do():
            if self._clients:
                await asyncio.gather(*[c.send(data) for c in self._clients],
                                     return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(_do(), self._loop)
        except Exception:
            pass

    # ── main.py 가 쓰는 인터페이스 ──
    def set_expression(self, label: str):
        """label: 기쁨 | 슬픔 | 위로 | 경청 | 놀람 | 평온  (+수면)"""
        self._cur_expr = label
        self._send({"cmd": "emotion", "name": label})

    def listen_start(self):
        """어르신 말을 듣기 시작했다 — 화면에 불을 켠다.

        꺼져 있다가 켜지는 순간 한 번 확 번쩍이고, 듣는 동안은 은은하게
        숨쉰다. 웨이크워드를 알아들었는지 멀리서도 바로 보이게 하려는 것이다.
        """
        self._listening = True
        self._send({"cmd": "listen_start"})

    def listen_stop(self):
        """다 들었다 — 불을 끈다(이제 생각하거나 말할 차례다)."""
        self._listening = False
        self._send({"cmd": "listen_stop"})

    def start_speaking(self):
        self._send({"cmd": "speak_start"})

    def stop_speaking(self):
        self._send({"cmd": "speak_stop"})

    def show_notice(self, title: str, sub: str = ""):
        """약 알림 같은 안내를 화면에 띄운다.

        말하는 동안만 얼굴을 덮는다. 소리를 놓쳤거나 잘 안 들리는 어르신도
        무슨 일인지 읽을 수 있게 하려는 것이다.
        """
        self._notice = (title, sub)
        self._send({"cmd": "notice", "title": title, "sub": sub})

    def hide_notice(self):
        """안내를 내리고 모리 얼굴로 돌아온다."""
        self._notice = None
        self._send({"cmd": "notice_hide"})

    def show_photo(self, url: str):
        """가족이 보낸 사진을 화면 전체에 띄운다."""
        self._send({"cmd": "photo", "url": url})

    def hide_photo(self):
        """사진을 내리고 얼굴로 복귀한다."""
        self._send({"cmd": "photo_hide"})

    def close(self):
        self._listening = False
        self._notice = None
        self._send({"cmd": "listen_stop"})
        self._send({"cmd": "notice_hide"})
        self._send({"cmd": "emotion", "name": "평온"})
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
