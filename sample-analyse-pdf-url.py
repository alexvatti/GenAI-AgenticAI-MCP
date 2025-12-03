from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Initialize client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

response = client.responses.create(
    model="gpt-5-mini",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Please read the attached PDF report and provide a concise summary. "
                        "Highlight the key insights, important metrics, and actionable recommendations."
                    )
                },
                {
                    "type": "input_file",
                    "file_url": "https://raw.githubusercontent.com/alexvatti/GenAI-AgenticAI-MCP/main/alex-report-06-Mar-2025-1764590459524.pdf.pdf"
                }
            ]
        }
    ]
)
print(response.output_text)
