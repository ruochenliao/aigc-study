from transformers import pipeline
if __name__ == "__main__":
    classifier = pipeline("sentiment-analysis")
    result = classifier("I love this product!")[0]
    print(f"label: {result['label']}, with score: {result['score']:.4f}")