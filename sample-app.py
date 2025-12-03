from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Read the key
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-5-mini",
    input="Write a short bedtime story about a unicorn."
)

print(response.output_text)
