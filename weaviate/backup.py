import weaviate as wv
import argparse
import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='filesystem')
    parser.add_argument('--id', type=str, default=None)
    parser.add_argument('--wait', type=bool, default=True)
        
    args, _ = parser.parse_known_args()
    
    
    # Backup
    auth_config = wv.auth.AuthApiKey(api_key="local-dev-wv-key")
    wv_client = wv.Client('http://localhost:8080', auth_client_secret=auth_config)

    if args.id:
        id = args.id
    else:
        id = datetime.datetime.now().astimezone().isoformat().replace(':','_').replace('.','_').replace('T','t').replace('+','plus')
        
    
    result = wv_client.backup.create(
        backup_id=id,
        backend=args.backend,
        wait_for_completion=args.wait
    )
    print(result)