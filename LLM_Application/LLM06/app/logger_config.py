"""
공용 로거 설정 (Day 2~4에서 만든 모듈의 최소 재현 버전)
"""
import logging


def setup_logger(name: str) -> logging.Logger:
    """이름별 로거를 반환합니다. (중복 핸들러 방지)"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
