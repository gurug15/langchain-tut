from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_nvidia import ChatNVIDIA
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

# llm =ChatNVIDIA(
#     model="qwen/qwen3.5-122b-a10b",
#     temperature=0.7
# )
llm =ChatNVIDIA(
    model="stepfun-ai/step-3.5-flash",
    temperature=0.7
)

connection  = sqlite3.connect(database="chatbot.db",check_same_thread=False)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = SqliteSaver(conn=connection)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)



def list_checkpoints():
    thread_set = set()
    for checkpoint in checkpointer.list(None):
        # print("Checkpoint:", checkpoint)
        thread_set.add(checkpoint.config['configurable']['thread_id'])
    return thread_set
