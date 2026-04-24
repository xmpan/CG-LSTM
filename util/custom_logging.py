import logging
from logging.handlers import RotatingFileHandler

# logging.basicConfig(level=logging.INFO,
#                     filename='./logfile/test.txt',
#                     filemode='a',
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# logging.info('This is begining')

class IntervalLogFilter(logging.Filter):
    def __init__(self, interval=2):
        self.interval = interval
        self.counter = 0

    def filter(self, record):
        self.counter += 1
        if (self.counter % self.interval == 0):
            self.counter = 0
            return True
        else:
            return False

logger = logging.getLogger('mylogger')

filter = IntervalLogFilter(1) # 每x次执行输出一次
logger.addFilter(filter)

# 配置logger
# log_file = './logfile/regression_one_attitute.log'
# log_file = './logfile/regression_multi_attitute.log'
# log_file = './logfile/regression_3out_withcnn.log'
# log_file = './logfile/regression_3out_fix_overfit.log'
# log_file = './logfile/regression_3out_with_unet.log'
# log_file = './logfile/regression_3out.log'
# log_file = './logfile/regression_2out_for_lossfcn.log' # test acc:88.55%
# log_file = './logfile/regression_2out_for_lossfcn_norepeat_data.log'
# log_file = './logfile/regression_2out_for_lossfcn_with_radarlos.log' # test acc:89.99%

# log_file = './logfile/transformerlog/transformer_0902-2.log'
# log_file = './logfile/lstmlog/3out/lstm_1013-1.log' # 2组姿态 base数据，没有归一化，没有处理平移敏感性
# log_file = './logfile/lstmlog/3out/lstm_1013-1-2.log' # 4组姿态
# log_file = './logfile/lstmlog/3out/lstm_1013-2.log' # 幅值归一化，没有平移敏感性处理
# log_file = './logfile/lstmlog/3out/lstm_1013-2-2.log' # 4组姿态
# log_file = './logfile/lstmlog/3out/lstm_1013-3.log'
# log_file = './logfile/lstmlog/3out/lstm_1013-3-2.log'

# log_file = './logfile/lstmlog/3out/lstm_1014-1.log'  # 6组姿态（2，3，4，5，6，7），base数据，输出姿态角
# log_file = './logfile/lstmlog/3out/lstm_1014-2.log'  # 6组姿态（2, 3, 4，5，6，7），base数据，输出四元数
# log_file = './logfile/lstmlog/3out/lstm_1014-3.log'    # 6组姿态 (2, 3, 4, 5, 6,7), base数据，输出theta-phi

# log_file = './logfile/lstmlog/3out/lstm_1201-lstm-se-unet-6d3-1.log'

# log_file = './logfile/lstmlog/output_compare/lstm_1101-lstm-euler2thetaphi-8d3.log'
log_file = './logfile/unet_omp_compare/1206_lstm_unet_se_6step_8d3_5snr-6.log' 


file_mode = 'a' # 追加模式
formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
handler = logging.FileHandler(log_file, mode=file_mode)
# 设计保存的日志文件大小（maxBytes=xxxx），以及日志文件数量（backupCount=5)
handler = RotatingFileHandler(log_file, maxBytes=50*1024*1024, backupCount=5)
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)
