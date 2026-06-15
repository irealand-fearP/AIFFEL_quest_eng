"""
요청 로깅 미들웨어 (Day 2~4에서 만든 모듈의 최소 재현 버전)
모든 요청의 메서드/경로/상태코드/처리시간을 로그로 남긴다.
"""
import time
from starlette.middleware.base import BaseHTTPMiddleware

from app.logger_config import setup_logger

logger = setup_logger("middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({elapsed_ms:.1f}ms)"
        )
        return response
