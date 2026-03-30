from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, START,END

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage,AIMessageChunk
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, EmailStr
from typing import TypedDict, Annotated, List

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import uvicorn
from dotenv import load_dotenv
import sqlite3
import requests

load_dotenv()


llm =ChatNVIDIA(
    model="deepseek-ai/deepseek-v3.1-terminus",
    temperature=0.7,
    streaming=True
)
app = FastAPI()

# router = APIRouter()
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def buy_stocks(symbol:str,quantity:int) -> str:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """

    decision = interrupt

    return {
        "status": "success",
        "message": f"Purchase order placed for {quantity} shares of {symbol}.",
        "symbol": symbol,
        "quantity": quantity,
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=JVZZBZSM8XCCQSSP"
    r = requests.get(url)
    return r.json()



tools = [search_tool, get_stock_price, calculator,buy_stocks]
llm_with_tools = llm.bind_tools(tools)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],

    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

tool_node = ToolNode(tools)
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state:ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

system_message = SystemMessage(content="You are a helpful assistant. dont provide any code, just answer the question in a concise manner. If you dont know the latest answer, say you dont know")

graph = StateGraph(ChatState)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)


class InputMessage(TypedDict):
    message:str


class ResponseMessage(TypedDict):
    id: int
    response: str







fake_db : List[ResponseMessage | InputMessage] = []


@app.get("/",response_model=List[ResponseMessage | InputMessage])
def get_messages():
    return fake_db



@app.post("/chat",response_model=ResponseMessage)
def sendMessage(inputMsg:InputMessage):
    fake_db.append(inputMsg)

    def generate():
        fullresponse = ""

        for message_chunk,metadata in chatbot.stream({"messages": [system_message, HumanMessage(inputMsg["message"])]},config={"configurable":{"thread_id":"user_1"}},stream_mode='messages'):
            if (
            isinstance(message_chunk, (AIMessage, AIMessageChunk))
            and message_chunk.content                        # not empty
            and not getattr(message_chunk, "tool_calls", []) # not a tool-call chunk
            and not getattr(message_chunk, "tool_call_chunks", [])
            ):
                chunk = message_chunk.content
                fullresponse += chunk
                yield chunk
        response:ResponseMessage = {
            "id": len(fake_db) +1,
            "response": fullresponse
        }
        fake_db.append(response)
    return StreamingResponse(generate(),media_type='text/plain')





if __name__ == "__main__":
    uvicorn.run("app:app",host="127.0.0.1",port=5555, reload=True)