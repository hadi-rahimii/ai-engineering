import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from langchain.agents import create_agent


def get_weather(city):
    ''' get the weather of a city .
    Args:
    city : a city name 
    '''
    return {'city': city, 'weather': 'sunny , 22C'}


dotenv.load_dotenv(override=True)

llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0.5)


class Agent:
    def __init__(self):
        self.llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0.5)
        self.messages = [SystemMessage(
            'give me short answer to the user question and use the provided tools when it is neccessary.')]

        self.agent = create_agent(model=self.llm, tools=[get_weather])

    def chat_loop(self):

        while True:
            user_input = input('ask question (type "q" for terminate ) :')
            if user_input.lower() == 'q':
                print('have a nice day')
                break

            self.messages.append(HumanMessage(user_input))
            response = self.agent.invoke({"messages": self.messages})
            self.messages.append(response['messages'][-1])
            print(response['messages'][-1].content)


def main():
    agent = Agent()
    agent.chat_loop()


if __name__ == '__main__':
    main()
