"""안전 판별 테스트. 오탐(멀쩡한 말을 위험으로 보는 것)을 특히 신경 써서 본다.

python test_safety.py 로 그냥 돌린다(인형에 pytest 를 깔지 않는다).
"""

import safety

OK = "\033[32m✓\033[0m"
NG = "\033[31m✗\033[0m"

fails = 0


def expect(text: str, kind, why: str = ""):
    global fails
    got = safety.classify(text)
    got_kind = got.kind if got else None
    good = got_kind == kind
    if not good:
        fails += 1
    mark = OK if good else NG
    detail = "" if good else f"  (나온 값: {got_kind})"
    print(f"  {mark} {text!r} → {kind}{detail}  {why}")


print("\n[자해·자살] 잡아야 하는 것")
expect("이제 그만 죽고 싶어", safety.SELF_HARM)
expect("차라리 죽었으면 좋겠어", safety.SELF_HARM)
expect("살기 싫다 이제", safety.SELF_HARM)
expect("영감 따라 죽어야지", safety.SELF_HARM)
expect("내가 없어져야 자식들이 편하지", safety.SELF_HARM)

print("\n[자해·자살] 걸리면 안 되는 것 — '죽겠다' 는 한국어에서 흔한 강조다")
expect("아이고 힘들어 죽겠네", None, "← 매일 나올 말")
expect("배고파 죽겠다", None)
expect("좋아 죽겠어 손주 온다니까", None)
expect("허리가 아파 죽겠어", None)

print("\n[자해·자살] 보조용언 '-어버리-' 가 껴도 어미가 기준이다")
expect("죽어버리고 싶어", safety.SELF_HARM, "← ~고 싶 : 욕구")
expect("그냥 죽어버렸으면 좋겠어", safety.SELF_HARM, "← ~었으면 : 소망")
expect("확 죽어버릴까", safety.SELF_HARM)
expect("힘들어 죽어버리겠네", None, "← ~겠 : 강조 관용구")
expect("아이고 죽어버리겠다", None)

print("\n[자해·자살] 체념조도 소극적 자살 사고로 본다")
expect("이제 죽어야지", safety.SELF_HARM)
expect("빨리 죽어야지 자식들 고생시키지 말고", safety.SELF_HARM)

print("\n[우선순위] '때려치우다' 는 폭력이 아니다")
expect("다 때려치우고 죽어버리고 싶다", safety.SELF_HARM, "← 학대로 새면 알림이 안 간다")
expect("일 때려치웠어", None)

print("\n[타해] 자해는 '죽어-', 타해는 '죽여-' — 한 글자 차이다")
expect("저 놈 죽여버리고 싶어", safety.HARM_OTHERS)
expect("확 죽여버릴까", safety.HARM_OTHERS)
expect("죽이고 싶을 만큼 밉다", safety.HARM_OTHERS)
expect("때려죽이고 싶어", safety.HARM_OTHERS)
expect("죽어버리고 싶어", safety.SELF_HARM, "← 한 글자 차이지만 자해다")

print("\n[타해] '죽여준다' 는 칭찬이다")
expect("이 노래 죽여준다", None)
expect("맛이 죽여주네", None)

print("\n[의료] 잡아야 하는 것")
expect("이 약 두 알 먹어도 되나?", safety.MEDICAL)
expect("약 안 먹어도 괜찮겠지?", safety.MEDICAL)
expect("병원 안 가도 될까", safety.MEDICAL)

print("\n[의료] 걸리면 안 되는 것")
expect("아침에 약 먹었어", None, "← 복용 보고일 뿐")
expect("약이 써서 싫어", None)

print("\n[학대] 잡아야 하는 것")
expect("며느리가 나를 때렸어", safety.ABUSE)
expect("밥을 안 주더라", safety.ABUSE)
expect("방에 가둬놨어", safety.ABUSE)

print("\n[학대] 걸리면 안 되는 것 — '맞았' 은 '옳다' 로 훨씬 자주 쓴다")
expect("네 말이 맞았어", None)
expect("어제 비가 많이 왔어", None)

print("\n[거친 말]")
expect("에이 씨발 리모컨이 어디 갔어", safety.PROFANITY)
expect("이런 지랄맞은 날씨가 있나", safety.PROFANITY)

print("\n[평범한 대화] 아무것도 걸리면 안 된다")
for t in ["오늘 날씨가 좋네", "손녀가 보고 싶어", "점심에 뭐 먹을까",
          "옛날에 부산 살았지", "다리가 좀 아파", "노래 한 곡 틀어줘",
          "우리 딸 이름이 뭐였더라", "집에 가고 싶어"]:
    expect(t, None)

print("\n[우선순위] 섞여 있으면 더 급한 쪽")
expect("죽고 싶은데 며느리가 밥도 안 줘", safety.SELF_HARM, "← 학대보다 자해가 먼저")

print("\n[빈 입력]")
expect("", None)
expect("   ", None)

print(f"\n{'모두 통과' if fails == 0 else str(fails) + '건 실패'}")
raise SystemExit(1 if fails else 0)
