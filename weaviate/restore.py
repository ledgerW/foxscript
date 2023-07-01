import weaviate as wv
import argparse
import datetime

from dotenv import dotenv_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str)
    parser.add_argument('--id', type=str, default=None)
    parser.add_argument('--classes', type=str, default=None)
    parser.add_argument('--wait', type=bool, default=False)
    parser.add_argument('--stage', default='local', type=str) 
    args, _ = parser.parse_known_args()
    
    # Connect to Weaviate
    config = dotenv_values(f'.env.{args.stage}')
    print('Connecting to {}'.format(config['WEAVIATE_URL']))

    auth_config = wv.auth.AuthApiKey(api_key=config['WEAVIATE_API_KEY'])
    wv_client = wv.Client(url=config['WEAVIATE_URL'], auth_client_secret=auth_config)
    schema = wv_client.schema.get()
    print([cl['class'] for cl in schema['classes']])

    if args.id:
        id = args.id
    else:
        id = datetime.datetime.now().astimezone().isoformat().replace(':','_').replace('.','_').replace('T','t').replace('+','plus')
        
    if args.classes:
        result = wv_client.backup.restore(
            backup_id=id,
            backend=args.backend,
            include_classes=args.classes.split(','),
            wait_for_completion=args.wait,
        )
    else:
        result = wv_client.backup.restore(
            backup_id=id,
            backend=args.backend,
            wait_for_completion=args.wait,
        )

    print(result)