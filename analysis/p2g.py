from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

model_dir = "kfkas/t5-large-korean-P2G"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# text = ["서규왕국 싸우디 태양광·풍녁 빨쩐 중심지 될 껃","서규왕국 싸우디 태양광·풍녁 빨쩐 중심지 될 껃"]
text = "서규왕국 싸우디 태양광·풍녁 빨쩐 중심지 될 껃"
inputs = tokenizer.encode(text,return_tensors="pt")
output = model.generate(inputs)
decoded_output = tokenizer.batch(output[0], skip_special_tokens=True)
print(decoded_output)#석유왕국 사우디 태양광·풍력 발전 중심지 될 것