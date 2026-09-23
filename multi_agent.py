import dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

dotenv.load_dotenv(override=True)

llm = ChatGroq(model='openai/gpt-oss-120b', temperature=0.4)


def fruit_info(fruit_name: str) -> str:
    """Look up info about a fruit."""
    print("🔥 FRUIT TOOL CALLED:" + fruit_name)
    facts = {
        "tomato": "Botanically a fruit (it develops from a flower and has seeds), "
        "but commonly treated as a vegetable in cooking.",
        "apple": "A pome fruit, related to roses. High in fiber and vitamin C.",
    }
    return facts.get(fruit_name.lower(), f"No specific info found for {fruit_name}.")


def veggie_info(veggie_name: str) -> str:
    """Look up info about a vegetable."""
    facts = {
        "carrot": "A root vegetable, high in beta-carotene (vitamin A).",
        "potato": "A starchy tuber vegetable, a staple food in many cuisines.",
    }
    return facts.get(veggie_name.lower(), f"No specific info found for {veggie_name}.")


fruit_agent = create_agent(
    model=llm,
    tools=[fruit_info],
    system_prompt="""
You are a fruit information agent.

For every fruit-related question, ALWAYS call the fruit_info tool.

IMPORTANT:
Your final answer MUST be based ONLY on the result returned by fruit_info.

You MUST NOT use your own knowledge.

If fruit_info returns:
"No specific info found for X."
then your final answer MUST say that no information was found for X.

Do not add any additional facts.

Respond in one sentence.
"""
)

veggie_agent = create_agent(
    model=llm,
    tools=[veggie_info],
    system_prompt='You are a veggie expert. Respond in one sentence.'
)


def ask_fruit_expert(question: str) -> str:
    """ask the fruit expert, use for all fruit quesrtions"""
    response = fruit_agent.invoke(
        {'messages': [{'role': 'user', 'content': question}]})
    return response['messages'][-1].content


def ask_veggie_expert(question: str) -> str:
    """ask the veggie expert, use for all veggie questions"""
    response = veggie_agent.invoke(
        {'messages': [{'role': 'user', 'content': question}]})
    return response['messages'][-1].content


main_agent = create_agent(
    model=llm,
    tools=[ask_fruit_expert, ask_veggie_expert],
    system_prompt='You have two experts: ask_fruit_expert and ask_veggie_expert.'
    'always delegate questions to the appropriate expert.'
)
print(fruit_info("banana"))

response = main_agent.invoke(
    {'messages': [{'role': 'user', 'content': 'what is banana ?'}]})
print(response["messages"][-1].content)
