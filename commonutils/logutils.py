import datetime
import logging
import daiquiri
from constants import commonconstants


def get_logger(name):
    daiquiri.setup(
        level=logging.DEBUG,
        outputs=(
            daiquiri.output.File(commonconstants.LOG_FILE_PATH, level=logging.DEBUG),
            daiquiri.output.TimedRotatingFile(
                commonconstants.LOG_FILE_PATH,
                level=logging.DEBUG,
                interval=datetime.timedelta(weeks=356))
        )
    )
    logger = daiquiri.getLogger(name)
    return logger
