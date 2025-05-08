# logger.py

import logging
import os

# Ensure the 'logs' directory exists
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Create a custom logger
logger = logging.getLogger('ChromaDB ORM Server')

# Set the global logging level (this will capture all messages from DEBUG and above)
logger.setLevel(logging.INFO)

# Define log formats
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a console handler (prints logs to console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # This will capture DEBUG and above
console_handler.setFormatter(log_format)

# Create a file handler (logs to a file)
file_handler = logging.FileHandler(os.path.join(log_directory, 'server.log'))
file_handler.setLevel(logging.DEBUG)  # This will capture DEBUG and above
file_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)