import dotenv
import os
from openai import OpenAI
from transformers import pipeline
dotenv.load_dotenv()


class Analyzer:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        self.messages = [{'role': 'system',
                          'content': '''Classify thr dentimental of the text into dance, work, or study.'''},]

    def sentiment_analysis(self, text):
        labels = ["dance", "study", "work"]
        return self.classifier(text, candidate_labels=labels)


def main():
    analyzer = Analyzer()
    while True:
        text = input(
            '''enter some text ( type 'q' for terminate) : ''').strip().lower()
        if text == 'q':
            print('good luck')
            break

        sentiment = analyzer.sentiment_analysis(text)
        print(sentiment['labels'])
        print(sentiment['scores'])


if __name__ == "__main__":
    main()
