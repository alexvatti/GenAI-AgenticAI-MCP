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
                    "text": "What teams are playing in this image?"
                },
                {
                    "type": "input_image",
                    "image_url": "https://raw.githubusercontent.com/alexvatti/GenAI-AgenticAI-MCP/main/LeBron_James_Layup.jpg"
                }
            ]
        }
    ]
)

print(response.output_text)
