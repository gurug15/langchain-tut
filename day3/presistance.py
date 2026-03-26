from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()


llm =ChatNVIDIA(
    model="qwen/qwen3.5-122b-a10b",
    temperature=0.7
)


class JokeState(TypedDict):

    topic: str
    joke: str
    explanation: str
def generate_joke(state: JokeState):

    prompt = f'generate a joke on the topic {state["topic"]}'
    response = llm.invoke(prompt).content

    return {'joke': response}
def generate_explanation(state: JokeState):

    prompt = f'write an explanation for the joke - {state["joke"]}'
    response = llm.invoke(prompt).content

    return {'explanation': response}

graph = StateGraph(JokeState)

graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

config1= {"configurable":{"thread_id":"1"}}
checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)



print(workflow.invoke({"topic":"samosa"},config=config1))

print(list(workflow.get_state_history(config1)))