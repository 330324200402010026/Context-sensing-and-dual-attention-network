# 数据集的类别
# NUM_CLASSES = 21
NUM_CLASSES = 30
#ucm 21  aid 30
# 训练时batch的大小
BATCH_SIZE = 64
VAL_BATCH_SIZE = 64

# 训练轮数
NUM_EPOCHS = 300

# 训练完成，精度和损失文件的保存路径,默认保存在trained_models下
TRAINED_MODEL = './logs/data_record3.pth'

# 数据集的存放位置
# TRAIN_DATASET_DIR = './dataset/train'
# VALID_DATASET_DIR = './dataset/val'

TRAIN_DATASET_DIR = '/tmp/pycharm_project_36/AIDdatasets55/train'
VALID_DATASET_DIR = '/tmp/pycharm_project_36/AIDdatasets55/val'
