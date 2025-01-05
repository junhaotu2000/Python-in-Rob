# main.py
from multiprocessing import Pool
from loguru import logger

def set_logger(logger_):
    global logger
    logger = logger_

def work(x):
    logger.info("Square rooting {}", x)
    return x**0.5

if __name__ == "__main__":
    logger.remove()
    logger.add("file.log", enqueue=True)

    with Pool(4, initializer=set_logger, initargs=(logger, )) as pool:
        results = pool.map(work, [1, 10, 100])

    logger.info("Done")