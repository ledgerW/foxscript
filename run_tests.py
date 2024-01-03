import os
from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    os.system("python -m unittest discover -s tests -v")