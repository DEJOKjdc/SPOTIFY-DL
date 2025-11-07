# test_hf_api.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env variables
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

# Initialize client with Hugging Face OpenAI endpoint
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_KEY
)

# Test prompt
prompt = "Return a JSON with a key 'test' and value 123."

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:groq",
        messages=[{"role": "user", "content": prompt}],
    )
    
    # Access content safely
    message_content = response.choices[0].message.content
    print("✅ API Response:")
    print(message_content)

except Exception as e:
    print("❌ HF OpenAI API call failed:", e)