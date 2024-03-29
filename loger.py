import logging

class Logger():
    def __init__(self,logPath):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(logPath)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(console)
        self.logger = logger

    def Print(self,message):
        self.logger.info(message)

        


