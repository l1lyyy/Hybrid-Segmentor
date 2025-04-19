import torch
import os 

# Huấn luyện từ tập dữ liệu đã copy từ Google Drive
NUM_EPOCHS = 100
DATASET_SIZE = {'train' : 4800, 'val' : 600, 'test' : 600}
# dataset = "/kaggle/input/crackvision6000/split_dataset_6000/"

# Đảm bảo đường dẫn này trỏ đến vị trí chính xác của dữ liệu trên hệ thống của bạn
dataset = r"C:\Users\minhn\Downloads\split_dataset_final" + "/"

#dataset = "C/Users/minhn/Downloads/split_dataset_final/"

#LEARNING_RATE = 1e-4
LEARNING_RATE = 5e-5  # Reduce learning rate for fine-tuning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 2
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

TRAIN_IMG_DIR = dataset+"train/IMG"
TRAIN_MASK_DIR = dataset+"train/GT"
VAL_IMG_DIR = dataset+"val/IMG"
VAL_MASK_DIR = dataset+"val/GT"
TEST_IMG_DIR = dataset+"test/IMG"
TEST_MASK_DIR = dataset+"test/GT"