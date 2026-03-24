from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, START,END

from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, EmailStr
from typing import TypedDict, Annotated, List

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import uvicorn
from dotenv import load_dotenv


load_dotenv()


llm =ChatNVIDIA(
    model="stepfun-ai/step-3.5-flash",
    temperature=0.7,
    streaming=True
)
app = FastAPI()

# router = APIRouter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],

    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state:ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


checkpointer = InMemorySaver()

system_message = SystemMessage(content="You are a helpful assistant. dont provide any code, just answer the question in a concise manner. If you dont know the answer, say you dont know")

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

    def generate():
        fullresponse = ""

        for messae_chunk,metadata in chatbot.stream({"messages": [system_message, HumanMessage(inputMsg["message"])]},config={"configurable":{"thread_id":"user_1"}},stream_mode='messages'):
            if messae_chunk.content:
                chunk = messae_chunk.content
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