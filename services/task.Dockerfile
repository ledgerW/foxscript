#FROM python:3.10-slim as build
FROM public.ecr.aws/lambda/python:3.10 as build
RUN yum install -y unzip && \
  curl -Lo "/tmp/chromedriver.zip" "https://chromedriver.storage.googleapis.com/113.0.5672.63/chromedriver_linux64.zip" && \
  curl -Lo "/tmp/chrome-linux.zip" "https://www.googleapis.com/download/storage/v1/b/chromium-browser-snapshots/o/Linux_x64%2F1121454%2Fchrome-linux.zip?alt=media" && \
  unzip /tmp/chromedriver.zip -d /opt/ && \
  unzip /tmp/chrome-linux.zip -d /opt/


FROM public.ecr.aws/lambda/python:3.10
RUN yum install atk cups-libs gtk3 libXcomposite alsa-lib \
  libXcursor libXdamage libXext libXi libXrandr libXScrnSaver \
  libXtst pango at-spi2-atk libXt xorg-x11-server-Xvfb \
  xorg-x11-xauth dbus-glib dbus-glib-devel -y

# Copy the requirements.txt file to the container
COPY task/Docker.requirements.txt Docker.requirements.txt

# Install the python requirements from requirements.txt
RUN python3.10 -m pip install -r Docker.requirements.txt

COPY --from=build /opt/chrome-linux /opt/chrome
COPY --from=build /opt/chromedriver /opt/

# Copy source code
RUN mkdir utils
COPY utils ./utils

COPY task/hollywood_writer.py hollywood_writer.py

# Override default lambda entrypoint
ENTRYPOINT [ "python3.10" ]

# Set the CMD to your handler
CMD ["hollywood_writer.py"]
