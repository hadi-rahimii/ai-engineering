import dotenv
import os
import time
from openai import OpenAI
from transformers import AutoTokenizer

dotenv.load_dotenv()


class ChatBot:
    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv('GROQ_API_KEY'),
            base_url="https://api.groq.com/openai/v1"

        )
        self.messages = [{'role': 'system',
                          'content': '''you are a helpful guy and u should always answer the users question'''}]
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.iteration = 0

    def bot(self, message):
        self.messages.append({"role": "user", "content": message}),
        response = self.llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=self.messages,
            temperature=0.7,
            stream=True
        )

        # response_text = response.choices[0].message.content
        # self.messages.append({"role": "assistant", "content": response_text})

        response_text = ''
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta.content

                if delta:
                    print(delta, end='', flush=True)
                    response_text += delta
                    time.sleep(0.05)

                if chunk.usage:
                    self.total_tokens += chunk.usage.total_tokens
                    self.input_tokens += chunk.usage.prompt_tokens
                    self.output_tokens += chunk.usage.completion_tokens
        print()
        self.messages.append(
            {'role': 'assistant', 'content': response_text})
        self.summarize_messages()
        return response_text

    def chat(self):

        encoding = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        while True:
            self.iteration += 1
            user_input = input(
                "Ask something (type 'q' for terminate):  ").strip().lower()

            if user_input == 'q':
                print("Exiting the chat. Goodbye!")
                break
            tokens_count = len(encoding.encode(user_input))

            # print(f"token count : {tokens_count}")
            # print(f'input tokens: {tokens_count}')

            if tokens_count > 1000:
                print('your prompt exceeded the limit , please write less ')
                continue

            if self.total_tokens > 10000:
                print('your prompt exceeded the limit , please try a hout later')
                break

            self.bot(user_input)
            # print(bot_answer)

    def trim_messages(self, max_number=4):
        if len(self.messages) < max_number:
            return
        system_messages = [m for m in self.messages if m['role'] == 'system']
        other_messages = [m for m in self.messages if m['role'] != 'system']

        trimmed_messages = other_messages[-max_number:]
        self.messages = [*system_messages, *trimmed_messages]

    def summarize_messages(self):
        if self.iteration % 5 != 0:
            return
        system_messages = [m for m in self.messages if m['role'] == 'system']
        other_messages = [m for m in self.messages if m['role'] != 'system']

        early_messages, last_messages = other_messages[0:5], other_messages[5:]

        response = self.llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {'role': 'system', 'content': 'Summarize these messages concisely, keeping key facts and information'}, *early_messages],
            temperature=0.5,
        )
        print(f"the summarised message : {response.choices[0].message.content}")
        summarized_messages = {
            'role': 'system', 'contect': f'this is  a summary of 5 early messages: {response.choices[0].message.content}'}

        self.messages = [*system_messages, summarized_messages, *last_messages]
        print(f'all messages {self.messages}')
        print('summarize was successfu;')

    def report(self):
        
        # for message in self.messages:
        #     print(message)
        print(f'input tokens: {self.input_tokens}')
        print(f'output tokens: {self.output_tokens}')
        print(f'total tokens: {self.total_tokens}')
        print(
            f'$: {(self.input_tokens * 0.05 + self.output_tokens * 0.08) / 1_000_000}'
        )
        # print(self.messages)


def main():
    bot = ChatBot()
    bot.chat()
    bot.report()


if __name__ == "__main__":
    main()
