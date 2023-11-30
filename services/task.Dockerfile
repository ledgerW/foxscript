FROM public.ecr.aws/lambda/python:3.10

# Copy the requirements.txt file to the container
COPY task/Docker.requirements.txt Docker.requirements.txt

# Install the python requirements from requirements.txt
RUN python3.10 -m pip install -r Docker.requirements.txt

# Copy source code
RUN mkdir utils
COPY utils ./utils

COPY task/run_batch_workflow.py run_batch_workflow.py

# Override default lambda entrypoint
ENTRYPOINT [ "python3.10" ]

# Set the CMD to your handler
CMD ["run_batch_workflow.py"]
