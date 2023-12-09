FROM public.ecr.aws/lambda/python:3.10

# Copy the requirements.txt file to the container
COPY services/task/Docker.requirements.txt Docker.requirements.txt

# Install the python requirements from requirements.txt
RUN python3.10 -m pip install -r Docker.requirements.txt

# Copy source code
RUN mkdir utils
COPY services/utils ./utils

RUN mkdir weaviate
COPY weaviate ./weaviate

COPY services/task/run_batch_workflow.py run_batch_workflow.py
COPY services/task/run_batch_upload_to_s3.py run_batch_upload_to_s3.py

# Override default lambda entrypoint
ENTRYPOINT [ "python3.10" ]

# Set the CMD to your handler
CMD ["run_batch_workflow.py"]
