from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model & tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Labels order from model card / config
labels = ["negative", "neutral", "positive"]

def fast_sentiment(sentences):
    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**tokens).logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    
    results = []
    for prob_dist in probs:
        prob_dict = {label: prob.item() for label, prob in zip(labels, prob_dist)}
        top_label = max(prob_dict, key=prob_dict.get)
        results.append({"sentiment": top_label, "probabilities": prob_dict})
    return results

if __name__ == "__main__":
    texts = [
    "Google unveiled innovative AI tools aimed at improving productivity across industries.",
    "Netflix subscriber growth slowed down this quarter, disappointing analysts.",
    "Intel announced plans to build new semiconductor factories in Europe to boost chip supply.",
    "Twitter faced backlash after changes to its content moderation policies sparked debates.",
    "Salesforce reported record-high revenue, driven by strong demand for cloud software."
    ]
    
    results = fast_sentiment(texts)
    
    for text, res in zip(texts, results):
        print(f"Text: {text}")
        print(f"Sentiment: {res['sentiment']}")
        print(f"Probabilities: {res['probabilities']}")

    # Calculate aggregate percentages
    from collections import Counter
    sentiments = [r['sentiment'] for r in results]
    counts = Counter(sentiments)
    total = len(sentiments)
    percentages = {k: (v / total) * 100 for k, v in counts.items()}
    print("Aggregate sentiment percentages:")
    for sentiment, pct in percentages.items():
        print(f"  {sentiment}: {pct:.2f}%")
