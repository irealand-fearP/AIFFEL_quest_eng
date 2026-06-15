"""
전역 에러 핸들러 (Day 2~4에서 만든 모듈의 최소 재현 버전)
모든 에러를 일관된 JSON 형식으로 반환하고, 적절한 상태 코드를 유지한다.
"""
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


def register_error_handlers(app):
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        # 입력 검증 실패 → 422 (필드 누락·범위 초과·타입 오류 등)
        # 원본 input 값에는 bytes 등 JSON 직렬화가 안 되는 값이 올 수 있어
        # 직렬화 가능한 type/loc/msg 만 추려서 담는다.
        detail = [
            {"type": e.get("type"), "loc": list(e.get("loc", [])), "msg": e.get("msg")}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"success": False, "error": "입력 검증 실패", "detail": detail},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "error": str(exc.detail)},
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        # 예상 못한 에러도 500으로 감싸 서버가 죽지 않게 한다
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "서버 내부 오류"},
        )
