from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# Load Pre-trained Model and Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

def summarize_text(text):
    if not text.strip():
        return "No text provided for summarization."
    
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)
    
    summary_ids = model.generate(inputs, max_length=150, min_length=30, num_beams=4, early_stopping=True)
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    input_text = ""
    
    if request.method == "POST":
        input_text = request.form.get("input_text")
        if input_text:
            summary = summarize_text(input_text)
    
    return render_template("index.html", input_text=input_text, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
