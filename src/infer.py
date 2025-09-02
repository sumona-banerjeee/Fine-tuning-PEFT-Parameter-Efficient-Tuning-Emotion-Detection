import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


MODEL_TYPE = "full"  
MODEL_PATH = "../results/full" if MODEL_TYPE == "full" else "../results/lora"

device = torch.device("cpu")


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if MODEL_TYPE == "full":
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
else:
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=5
    )
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)

model.to(device)
model.eval()

id2label = {0: "joy", 1: "anger", 2: "sadness", 3: "fear", 4: "surprise"}

print("Emotion Detection (type 'q')")
while True:
    text = input("Enter your sentence: ").strip()
    if text.lower() in ["q"]:
        break
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    print(f"Emotional category: {id2label[pred]}\n")
