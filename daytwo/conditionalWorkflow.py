from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import re

load_dotenv()

model = ChatNVIDIA(
    model="qwen/qwen3.5-122b-a10b",
    temperature=0.7
)

llm = ChatOllama(model="ministral-3")


class SentimentSchema(BaseModel):
    sentiment: Literal["positive","negative"] = Field(description="sentiment of the review")


class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive","negative"]
    diagnosis: dict
    response: str

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')


structured_model = llm.with_structured_output(SentimentSchema)

structured_model2 = llm.with_structured_output(DiagnosisSchema)

def find_sentiment(state:ReviewState):
    prompt=f"for the following review find its sentiment \n {state["review"]}"
    sentiment = structured_model.invoke(prompt).sentiment

    return {"sentiment":sentiment}

# check sentiment this part is which makes it conditional workflow
def check_sentiment(state:ReviewState)-> Literal["positive_response","run_diagnosis"]:
    if state["sentiment"] == 'positive':
        return "positive_response"
    else:
        return "run_diagnosis"
    
def positive_response(state:ReviewState):
    prompt = f"""Write a warm thank-you message in response to this review:
    \n\n\"{state['review']}\"\n
Also, kindly ask the user to leave feedback on our website."""
    
    response = model.invoke(prompt).content

    return {'response': response}
    
def run_diagnosis(state:ReviewState):
    prompt = f"""Diagnose this negative review:\n\n{state['review']}\n"
    "Return issue_type, tone, and urgency.
"""
    response = structured_model2.invoke(prompt)

    return {'diagnosis': response.model_dump()}


def negative_response(state: ReviewState):

    diagnosis = state['diagnosis']

    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message.
"""
    response = model.invoke(prompt).content

    return {'response': response}


graph = StateGraph(ReviewState)


graph.add_node('find_sentiment', find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negative_response', negative_response)

graph.add_edge(START, 'find_sentiment')

graph.add_conditional_edges("find_sentiment",check_sentiment)

graph.add_edge('positive_response', END)

graph.add_edge('run_diagnosis', 'negative_response')
graph.add_edge('negative_response', END)


workflow = graph.compile()



print(workflow.invoke({"topic","what is wrong with your service, how do i fix it now"}))