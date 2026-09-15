import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
dotenv.load_dotenv()


class Agent:
    def __init__(self):
        self.llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0.5)

    def chat(self):
        
        #///////////////////////
        # creating template for almost static prompt
        
        # template = '''
        # Dear {user_name},
        # Appologizing for the bad expreince you had with our service. We are committed to improving and ensuring that such issues do not occur in the future. Your feedback is invaluable to us, and we appreciate your patience and understanding.
        # Sincerely,
        # {brand_name}
        
        # '''
        # prompt = PromptTemplate(template=template, input_variables=['user_name', 'brand_name'])
        # print(prompt.format(user_name='Hadi', brand_name='Tech Solutions Inc.'))
        
        #///////////////////////////////////////
        # usin chain for runnable prompt with llm
        
        prompt = ChatPromptTemplate.from_messages([{'role':"system",'content':'you are a translator from {origin} to {to}'},{'role':'user','content':'{input}'}])
        
        chain = prompt | self.llm
        response = chain.invoke({"origin":'English', "to":'Persian', "input":'hello how are?'})
        print(response.content)


def main():

    agent = Agent()
    agent.chat()


if __name__ == '__main__':
    main()
