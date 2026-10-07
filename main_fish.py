"""
main_fish.py
------------
Fish Audio TTS 테스트용 실행 파일. main.py 와 같은 파이프라인을 TTS 만 Fish 로 바꿔 돌린다.

    python main_fish.py

.env 에 FISH_API_KEY 만 있으면 된다. 기존 main.py 는 그대로 CosyVoice 로 동작한다.
"""

import os

from dotenv import load_dotenv

# main 을 import 하기 전에 정해야 한다(main.py 가 import 시점에 TTS_ENGINE 을 읽음).
# .env 를 먼저 읽어서, .env 에 FISH_REFERENCE_ID 가 있으면 그 값이 우선한다.
load_dotenv()
os.environ["TTS_ENGINE"] = "fish"
os.environ.setdefault("FISH_REFERENCE_ID", "c2045f5c866b4222a4a2c30b5b051f7e")

import main  # noqa: E402

if __name__ == "__main__":
    main.main()
