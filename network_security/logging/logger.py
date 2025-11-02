import logging
import os
from datetime import datetime

log_file_format = f'{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'
log_file_path = os.path.join(os.getcwd(), 'logs', log_file_format)
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)