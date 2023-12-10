import subprocess
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stage', default='dev', type=str)
    args, _ = parser.parse_known_args()

    base = "serverless-offline-multi"
    api = "--directory api --port 3005 --watch true"
    data = "--directory data --port 3006 --watch true"
    task = "--directory task --port 3007 --watch true"
    backend_jobs = "--directory backend-jobs --port 3008 --watch true"
    stage = f"--stage {args.stage}"

    cmd = ' '.join([base, api, data, task, backend_jobs, stage])

    subprocess.run(cmd, shell=True) 