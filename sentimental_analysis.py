import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()


class Analyzer:
    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv('GROQ_API_KEY'),
            base_url="https://api.groq.com/openai/v1"

        )
        self.messages = [{'role': 'system',
                          'content': '''Classify thr dentimental of the text into positive, negative, or neutral. Provide a brief explanation for your classification.'''}]

    def sentiment_analysis(self, text):

        response = self.llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[*self.messages, {'role': 'user', 'content': text}],
            temperature=0.7,
            max_tokens=100,
        )
        return response.choices[0].message.content


def main():
    analyzer = Analyzer()
    sentiment = analyzer.sentiment_analysis(
        "I love this product! It's amazing and exceeded my expectations.")
    print("Sentiment Analysis Result:")
    print(sentiment)


if __name__ == "__main__":
    main()
