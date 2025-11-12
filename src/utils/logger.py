import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Handlers
file_logs = logging.FileHandler("./logs/log.log")
console_logs = logging.StreamHandler()

# Formatter
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
file_logs.setFormatter(formatter)
console_logs.setFormatter(formatter)

# Handlers
logger.addHandler(file_logs)
logger.addHandler(console_logs)
