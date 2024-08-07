# Pull the base image with python 3.10 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.11 as build
RUN yum install -y unzip && \
  curl -Lo "/tmp/chromedriver.zip" "https://chromedriver.storage.googleapis.com/113.0.5672.63/chromedriver_linux64.zip" && \
  curl -Lo "/tmp/chrome-linux.zip" "https://www.googleapis.com/download/storage/v1/b/chromium-browser-snapshots/o/Linux_x64%2F1121454%2Fchrome-linux.zip?alt=media" && \
  unzip /tmp/chromedriver.zip -d /opt/ && \
  unzip /tmp/chrome-linux.zip -d /opt/


FROM public.ecr.aws/lambda/python:3.11
RUN yum install atk cups-libs gtk3 libXcomposite alsa-lib \
  libXcursor libXdamage libXext libXi libXrandr libXScrnSaver \
  libXtst pango at-spi2-atk libXt xorg-x11-server-Xvfb \
  xorg-x11-xauth dbus-glib dbus-glib-devel -y

# Copy the requirements.txt file to the container
COPY services/data/Docker.requirements.txt ./

# Install the python requirements from requirements.txt
RUN python3.11 -m pip install -r Docker.requirements.txt

RUN mkdir nltk_data
ENV NLTK_DATA="nltk_data"
RUN python3.11 -m nltk.downloader -d nltk_data all

COPY --from=build /opt/chrome-linux /opt/chrome
COPY --from=build /opt/chromedriver /opt/

# Copy lambda source code
RUN mkdir utils
COPY services/utils ./utils

RUN mkdir scrapers
COPY services/data/scrapers ./scrapers

RUN mkdir weaviate
COPY weaviate ./weaviate

COPY services/data/load_data.py ./
COPY services/data/researcher.py ./
COPY services/data/keyword_doc_checker.py ./
COPY services/data/news_sources.txt ./
COPY services/task/ecs_api.py ./
COPY services/task/run_task_keyword_planner.py ./

# Set the CMD to your handler
CMD ["data.master"]
