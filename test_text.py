import requests

url = "http://127.0.0.1:8000/v1/chat/completions"
body = {
    "model": "qwen3vl_lora_local",
    "messages": [
        {
            "role": "system",
            "content": "你是智能体\n\n【如果是隐患排查】..."
        },
        {
            "role": "user",
            "content": "用户问题：你是谁"
        }
    ],
    "temperature": 0.1,
}
resp = requests.post(url, json=body)
print(resp.status_code, resp.text)
