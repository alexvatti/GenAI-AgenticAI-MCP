import os
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Initialize client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Create response
response = client.responses.create(
    model="gpt-5-mini",
    instructions="You are a mutual fund assistant. Provide clear, simple financial guidance.",
    input="Explain the difference between small cap and mid cap mutual funds."
)

# Print output
print(response.output_text)