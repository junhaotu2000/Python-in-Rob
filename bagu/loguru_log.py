#%%
from loguru import logger

# %%
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')

# %%
logger.info('this is {}', "a info log")
logger.info('this is {}', 3)
logger.info('this is {:.3f}', 3.14)

# %%
logger.add('log.log', level='INFO', mode='w')
logger.info('this is a info log')

# %%
logger.add('log2.log', 
           level='INFO',
           rotation='10 KB')

for i in range(100):
    logger.debug('This is {} debug message', i)
    logger.info('This is {} info message', i)
    logger.warning('This is {} warning message', i)
    logger.error('This is {} error message', i)
    logger.critical('This is {} critical message', i)
