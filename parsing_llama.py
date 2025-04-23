import os
import json
from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
import re

# Load environment variables
load_dotenv()

# Set up LlamaParse parser
parser = LlamaParse(result_type="markdown")  # Change to "text" if needed

# Define the folder containing PDFs
input_folder = "/Users/kevinsmith/Documents/pdf test"

# Use SimpleDirectoryReader to parse all PDFs in the folder
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_dir=input_folder, file_extractor=file_extractor).load_data()

# Function to extract tables from markdown text
def extract_tables(text):
    tables = []
    table_pattern = re.compile(r"(\|.+\|(?:\n\|[-:]+[-|:]+\|)?(?:\n\|.+\|)*)", re.MULTILINE)

    for i, match in enumerate(table_pattern.findall(text)):
        lines = match.strip().split("\n")
        headers = lines[0].strip("|").split("|")  # Extract headers
        rows = [line.strip("|").split("|") for line in lines[2:]]  # Extract row data
        tables.append({
            "table_number": i + 1,
            "headers": [h.strip() for h in headers],
            "rows": [[cell.strip() for cell in row] for row in rows]
        })

    return tables

# Function to extract metadata from markdown (basic approach)
def extract_metadata(text):
    metadata = {"title": None, "author": None, "date": None}

    # Extract title (assuming first heading is the title)
    title_match = re.search(r"^# (.+)", text, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()

    # Extract author and date (if found in first few lines)
    author_match = re.search(r"(?i)author: (.+)", text)
    date_match = re.search(r"(?i)date: (\d{4}-\d{2}-\d{2})", text)

    if author_match:
        metadata["author"] = author_match.group(1).strip()
    if date_match:
        metadata["date"] = date_match.group(1).strip()

    return metadata

# Process and store structured data
parsed_data = []
for doc in documents:
    content = doc.text

    # Extract metadata
    metadata = extract_metadata(content)

    # Extract tables and remove them from main text
    tables = extract_tables(content)
    content_cleaned = re.sub(r"(\|.+\|(?:\n\|[-:]+[-|:]+\|)?(?:\n\|.+\|)*)", "", content).strip()

    # Store structured data
    parsed_data.append({
        "file_name": doc.metadata.get("file_name", "unknown"),
        "metadata": metadata,
        "body": content_cleaned,
        "tables": tables
    })

# Save structured content to JSON
output_json_path = "parsed_documents_structured.json"
with open(output_json_path, "w", encoding="utf-8") as json_file:
    json.dump(parsed_data, json_file, ensure_ascii=False, indent=4)

print(f"Structured parsed content saved to {output_json_path}")

