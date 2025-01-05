import logging

# 设置日志级别为INFO及以上(全局)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='log.log',
                    filemode='w') 

# other_logger = logging.getLogger(__name__)
test_logger = logging.getLogger('test')


logging.debug('This is a debug message')
logging.info('This is an info message')
file_handler = logging.FileHandler('test_log.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
test_logger.addHandler(file_handler)

'''
在默认情况下，logging模块将日志打印到屏幕上，
日志级别为WARNING和Warning级别以上的日志信息才会被打印出来
'''
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')

test_logger.info('This is a test info message')
try: 1/0
except Exception as e:
    # test_logger.expception('An exception occurred') # 会报错在terminal上
    test_logger.error('This is an error message', exc_info=True) # 会打印出异常信息在log文件中