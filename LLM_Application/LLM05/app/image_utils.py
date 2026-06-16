"""
이미지 업로드 안전장치 + 전처리
"""
from fastapi import UploadFile, HTTPException
from PIL import Image
import io

# 허용 설정
ALLOWED_TYPES = {"image/png", "image/jpeg"}   # 표준 MIME만 (.jpg도 브라우저는 image/jpeg로 전송)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


async def validate_and_read_image(
    file: UploadFile,
    max_size: int = MAX_FILE_SIZE,
    target_size: tuple = (28, 28),
) -> Image.Image:
    """
    업로드된 파일을 검증하고, PIL 이미지로 반환합니다.

    검증 순서:
      1. 파일 타입 검증 → 허용된 형식(PNG, JPEG)만 통과
      2. 파일 크기 검증 → 5MB 이하만 통과
      3. 이미지 디코딩 검증 → 실제로 열 수 있는 이미지만 통과
      4. 리사이징 + 그레이스케일 변환 → 모델 입력 크기에 맞춤
    """
    # ─── 1. 파일 타입 검증 (1차 거름망) ─────────────
    # content_type은 클라이언트가 보낸 MIME 타입이라 위조될 수 있습니다.
    # 명백히 엉뚱한 형식(text/plain 등)을 싸게 걸러내는 1차 방어이고,
    # 위장 파일의 진짜 차단은 3단계(PIL 디코딩)가 담당합니다.
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다: {file.content_type}. "
                   f"허용 형식: {ALLOWED_TYPES}",
        )

    # ─── 2. 파일 크기 검증 ─────────────────────────
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"파일 크기가 {max_size // (1024*1024)}MB를 초과합니다. "
                   f"현재: {len(contents) / (1024*1024):.1f}MB",
        )

    # ─── 3. 이미지 디코딩 검증 ─────────────────────
    # content_type이 image/png여도 내용이 실제 이미지가 아닐 수 있으므로 PIL로 확인합니다.
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="이미지를 읽을 수 없습니다. 파일이 손상되었을 수 있습니다.",
        )

    # ─── 4. 리사이징 + 그레이스케일 변환 ──────────────
    image = image.convert("L").resize(target_size)

    return image
