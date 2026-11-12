import boto3
import json
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-west-2")

messages = [
    {
        "role": "user",
        "content": [
            {
                "text": "What is Amazon Bedrock?"  # TODO: Add your prompt here (e.g., "What is Amazon Bedrock?")
            }
        ]
    }
]

inf_params = {
    "maxTokens": 1024,
    "topP": 0.1,
    "temperature": 0.1
}

native_request = {
    "messages": messages,
    "inferenceConfig": inf_params,
}

request = json.dumps(native_request)

try:
    response = client.invoke_model(modelId='us.amazon.nova-lite-v1:0', body=request)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke 'us.amazon.nova-lite-v1:0'. Reason: {e}")
    exit(1)

model_response = json.loads(response["body"].read())
response_text = model_response["output"]["message"]["content"][0]["text"]
print(response_text)

print("-" * 50)
print("Now enter this in the answer textbox at the top of the task on the AWS Jam page:")
print(json.dumps(inf_params))