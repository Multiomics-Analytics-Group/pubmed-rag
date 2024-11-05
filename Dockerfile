# Use the official Python image from the Docker Hub
FROM python:3.11

# Copy the module folder into the Docker image
COPY pubmed_rag /app/pubmed_rag

# If you have a requirements.txt, copy and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set the working directory
WORKDIR /app

# Download the decided model, faster
RUN python -c \
"from transformers import AutoModel, AutoTokenizer; \
AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext');\
AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')"

CMD ["python"]