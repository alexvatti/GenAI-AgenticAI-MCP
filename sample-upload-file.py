from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Initialize client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

file = client.files.create(
    file=open(r"D:\GenAI-AI-AGENT\Resourses\alex-report-06-Mar-2025-1764590459524.pdf.pdf", "rb"),
    purpose="user_data"
)

response = client.responses.create(
    model="gpt-5-mini",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": file.id,
                },
                {
                    "type": "input_text",
                    "text": "What is the HbA1C level in the file?",
                },
            ]
        }
    ]
)

print(response.output_text)