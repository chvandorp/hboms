import logging

logger = logging.getLogger("HBOMS")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(handler)
