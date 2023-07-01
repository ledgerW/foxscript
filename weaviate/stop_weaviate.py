import os
import subprocess
from time import sleep

import weaviate as wv


if __name__ == "__main__":
  
  # start weaviate
  start_cmd = 'docker compose -f weaviate/docker-compose.yml down'
  
  subprocess.run(
    start_cmd,
    env=dict(os.environ),
    capture_output=True,
    shell=True
  )