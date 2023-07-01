import argparse
import os
import subprocess
import datetime
from time import sleep

import weaviate as wv

from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--noaws', default=True, type=bool)
  parser.add_argument('--backend', default='filesystem', type=str)
  parser.add_argument('--id', type=str, default=None)
  parser.add_argument('--classes', type=str, default=None)
  parser.add_argument('--clean', action='store_true', default=False)
  parser.add_argument('--wait', action='store_true', default=True)
    
  args, _ = parser.parse_known_args()
  
  # start weaviate
  start_cmd = 'docker compose -f weaviate/docker-compose.yml up -d --wait'
  
  subprocess.run(
    start_cmd,
    env=dict(os.environ),
    capture_output=True,
    shell=True
  )

  sleep(2)

  # Restore
  auth_config = wv.auth.AuthApiKey(api_key="local-dev-wv-key")
  wv_client = wv.Client(
     url="http://localhost:8080",
     auth_client_secret=auth_config
    )


  if args.id: 
    if args.classes:
        result = wv_client.backup.restore(
            backup_id=args.id,
            backend=args.backend,
            include_classes=args.classes.split(','),
            wait_for_completion=args.wait,
        )
    else:
        result = wv_client.backup.restore(
            backup_id=args.id,
            backend=args.backend,
            wait_for_completion=args.wait,
        )

    print(result)