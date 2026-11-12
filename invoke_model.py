import sys
import json
import boto3
from botocore.exceptions import ClientError

# Usage: python3 invoke_model.py [temperature] [topP] [maxTokens]
temperature = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
top_p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

client = boto3.client("bedrock-runtime", region_name="us-west-2")

messages = [
    {
        "role": "user",
        "content": [{"text": "What is Amazon Bedrock?"}]
    }
]

inf_params = {
    "maxTokens": max_tokens,
    "topP": top_p,
    "temperature": temperature
}

native_request = {
    "messages": messages,
    "inferenceConfig": inf_params,
}

request = json.dumps(native_request)

try:
    sts = boto3.client("sts")
    print("Running as:", sts.get_caller_identity()["Arn"])
    response = client.invoke_model(modelId="us.amazon.nova-lite-v1:0", body=request)
except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke 'us.amazon.nova-lite-v1:0'. Reason: {e}")
    exit(1)

model_response = json.loads(response["body"].read())
response_text = model_response["output"]["message"]["content"][0]["text"]
print(response_text)

print("-" * 50)
print("Now enter this in the answer textbox at the top of the task on the AWS Jam page:")
print(json.dumps(inf_params))
