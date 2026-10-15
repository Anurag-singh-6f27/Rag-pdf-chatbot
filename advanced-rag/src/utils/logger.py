import logging


logger = logging.getLogger("advanced_rag")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)