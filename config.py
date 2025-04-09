from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

client = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)


def get_agent_client():
    return client
