import sys
sys.stdout.reconfigure(encoding='utf-8')
from google import genai
import api_server

client = genai.Client(api_key=api_server.GOOGLE_API_KEY)
models_to_test = ['gemini-2.5-flash-lite', 'gemini-flash-lite-latest', 'gemini-flash-latest', 'gemini-3-flash-preview']

for m in models_to_test:
    print(f"\n--- Testing {m} ---")
    try:
        response = client.models.generate_content(
            model=m,
            contents='Please analyze the text and output structural json as requested.',
            config={
                'response_mime_type': 'application/json',
                'response_schema': api_server.ApiQuantFeatures
            }
        )
        print(f"SUCCESS for {m}! Output generated.")
        print(f"Use model: '{m}'")
        break
    except Exception as e:
        print(f"FAILED for {m}: {str(e)}")
