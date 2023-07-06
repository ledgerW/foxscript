# Pull the base image with python 3.10 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.10

# Copy the requirements.txt file to the container
COPY api/Docker.requirements.txt ./

# Install the python requirements from requirements.txt
RUN python3.10 -m pip install -r Docker.requirements.txt

# Copy lambda source code
RUN mkdir utils
COPY utils ./utils

COPY api/weaviate_api.py ./
COPY api/workflow_api.py ./

# Set the CMD to your handler
CMD ["workflow_api.handler"]
