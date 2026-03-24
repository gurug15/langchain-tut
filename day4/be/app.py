from fastapi import FastAPI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, START,END

from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, EmailStr
from typing import TypedDict, Annotated, List

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import uvicorn
from dotenv import load_dotenv


load_dotenv()


llm =ChatNVIDIA(
    model="qwen/qwen3.5-122b-a10b",
    temperature=0.7
)
app = FastAPI()

# router = APIRouter()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state:ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


checkpointer = InMemorySaver();


graph = StateGraph(ChatState)

graph.add_node("chat_node",chat_node)


graph.add_edge(START, "chat_node")
graph.add_edge("chat_node",END)

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
    message = chatbot.invoke({
        "messages":HumanMessage(inputMsg["message"])
    },config={"configurable":{"thread_id":"user_1"}})["messages"][-1].content
    response:ResponseMessage  = {"id": len(fake_db) +1 , "response":message}
    fake_db.append(response)
    return response





if __name__ == "__main__":
    uvicorn.run("app:app",host="127.0.0.1",port=5555, reload=True)