import multiprocessing
from loguru import logger

def my_proces(logger_):
    logger_.info('This is a info message')
    logger_.complete()
    
    
if __name__ == '__main__':
    logger.remove()
    logger.add('log.log', enqueue=True)
    
    process = multiprocessing.Process(target=my_proces, args=(logger,))
    process.start()