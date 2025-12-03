

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Initialize client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
response = client.responses.create(
    model="gpt-5-mini",
    tools=[{"type": "web_search"}],
    input=(
        "Summarize the performance of small cap, mid cap, large cap, "
        "and sectoral mutual funds in India for the last 3 months, "
        "and provide any predictions or analyst views for the next 3 months."
    )
)

print(response.output_text)