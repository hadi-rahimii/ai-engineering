import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import os
from langchain.agents import create_agent
import requests
from langchain_tavily import TavilySearch

def get_weather(city: str):
    """
    Fetches the current weather information for a given city using the OpenWeatherMap API.

    Args:
        city (str): 

    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "API key not found. Please set OPENWEATHER_API_KEY in your .env file."}

    # آدرس پایه API برای آب و هوای فعلی
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    # پارامترهای درخواست
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",  # برای دریافت دما به سلسیوس
        "lang": "fa"        # برای دریافت توضیحات به فارسی (اختیاری)
    }

    try:
        # ارسال درخواست GET با timeout برای جلوگیری از معطلی [citation:12]
        response = requests.get(base_url, params=params, timeout=10)
        
        # اگر کد وضعیت خطا باشد، این خط یک Exception ایجاد می‌کند [citation:2]
        response.raise_for_status() 
        
        data = response.json()
        
        # استخراج اطلاعات مورد نیاز از پاسخ JSON
        weather_info = {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "condition": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
        }
        return weather_info

    except requests.exceptions.HTTPError as http_err:
        # این خطا زمانی رخ می‌دهد که شهر پیدا نشود یا API Key معتبر نباشد
        if response.status_code == 404:
            return {"error": f"City '{city}' not found."}
        return {"error": f"HTTP error occurred: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        # این خطا برای مسائل مربوط به شبکه (مثل قطعی اینترنت یا timeout) است [citation:2][citation:12]
        return {"error": f"Request failed: {req_err}"}

search_tool = TavilySearch(max_results= 1)

dotenv.load_dotenv(override=True)

llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0.5)


class Agent:
    def __init__(self):
        self.llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0.5)
        self.messages = [SystemMessage(
            'give me short answer to the user question and use the provided tools when it is neccessary.')]

        self.agent = create_agent(model=self.llm, tools=[get_weather, search_tool])

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
