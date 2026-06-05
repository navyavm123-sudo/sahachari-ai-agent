from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct"
)

response = pipe(
    "What is Python?",
    max_new_tokens=100
)

print(response[0]["generated_text"])