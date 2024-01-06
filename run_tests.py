import os
from dotenv import load_dotenv


if __name__ == "__main__":
    os.chdir('services/api')
    load_dotenv('.env')
    os.system("python -m unittest discover -s tests -v")