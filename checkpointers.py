import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
# from langchain.agents.middleware import SummarizationMiddlewared
from langgraph.checkpoint.memory import InMemorySaver

dotenv.load_dotenv(override=True)


class Agent:
    def __init__(self):
        self.llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0.5)

        self.agent = create_agent(model=self.llm, system_prompt='give shor answers', tools=[
        ], checkpointer=InMemorySaver())

    def chat_loop(self):
        active_session = 'session_1'
        while True:
            user_input = input(
                'ask something (type "q" to terminarte, type "ch" to change session) :').strip().lower()

            if user_input == 'q':
                print('good luck')
                break
            if user_input == 'ch':
                active_session = 'sesstion_2' if active_session == "session_1" else "session_1"
                print('Done')
                continue

            config = {'configurable': {'thread_id': active_session}}
            response = self.agent.invoke(
                {'messages': HumanMessage(user_input)}, config)

            print(response['messages'][-1].content)


def main():
    agent = Agent()
    agent.chat_loop()


if __name__ == '__main__':
    main()
