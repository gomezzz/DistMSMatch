import logging, os


def get_logger(name, save_path=None, level="INFO"):
    """Initializes the logger

    Args:
        name (str): logger name
        save_path (str, optional): path to save the log file. Defaults to None.
        level (str, optional): Logging level. Defaults to "INFO".

    Returns:
        Logger: the created logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, "log.txt"))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger
