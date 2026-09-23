import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder

from pydantic import BaseModel, Field
dotenv.load_dotenv()


class User(BaseModel):
    name : str = Field(description='name of the user')
    age: int = Field(description='age of the user')
    companies:list[str] = Field(description='the companies that related to the user')
    net_worth : str = Field(description='the net worth of the user')
    

class Agent:
    def __init__(self):
        self.llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0.5)
        
    def chat(self):

        # ///////////////////////
        # creating template for almost static prompt

        # template = '''
        # Dear {user_name},
        # Appologizing for the bad expreince you had with our service. We are committed to improving and ensuring that such issues do not occur in the future. Your feedback is invaluable to us, and we appreciate your patience and understanding.
        # Sincerely,
        # {brand_name}

        # '''
        # prompt = PromptTemplate(template=template, input_variables=['user_name', 'brand_name'])
        # print(prompt.format(user_name='Hadi', brand_name='Tech Solutions Inc.'))

        # ///////////////////////////////////////
        # usin chain for runnable prompt with llm

        # history = [HumanMessage(' hello '), AIMessage('!!! سلام !!!'), HumanMessage(
        #     ' how are you ?'), AIMessage('!!!  حالت چطوره ؟!!!')]

        # prompt = ChatPromptTemplate.from_messages([{'role': "system", 'content': 'you are a translator from {origin} to {to}'}, MessagesPlaceholder(
        #     variable_name="history_name"), {'role': 'user', 'content': '{input}'}])

        # chain = prompt | self.llm
        # response = chain.invoke(
        #     {"origin": 'English', "to": 'Persian', "history_name": history, "input": 'im fine'})
        # print(response.content)
        
        
        
        structured_llm =self.llm.with_structured_output(User, method='json_schema')
        response = structured_llm.invoke('Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is a businessman and former public official who is the CEO and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and briefly became the only trillionaire (in terms of US dollars) in June 2026; as of August 14, 2026, Forbes estimates his net worth to be US$864 billion.')
        print(response)
        
        


def main():

    agent = Agent()
    agent.chat()


if __name__ == '__main__':
    main()
