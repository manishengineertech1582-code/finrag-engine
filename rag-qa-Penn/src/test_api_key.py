import os

api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print("FULL API KEY:", api_key)  # only for private verification
else:
    print("Key not found!")