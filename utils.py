import yaml
import logging
import os
from datetime import datetime

# Helper Functions
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Configuration Management
def save_config(config, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(config, file)

# Common Operations
def read_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def write_data(data, file_path):
    with open(file_path, 'w') as file:
        file.write(data)

# Example Usage
if __name__ == "__main__":
    # Logging Setup
    logger = setup_logging()
    
    # Configuration Management
    config = load_config('config.yaml')
    logger.info(f"Loaded configuration: {config}")
    
    # Save updated configuration
    new_config = {'log_level': 'DEBUG', 'threshold': 5}
    save_config(new_config, 'config.yaml')
    
    # Common Operations
    data = read_data('data.txt')
    logger.info(f"Read data: {data}")
    
    new_data = f"Data at {datetime.now()}"
    write_data(new_data, 'data.txt')