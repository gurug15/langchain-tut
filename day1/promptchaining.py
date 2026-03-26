from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_ollama  import ChatOllama
from typing import TypedDict
from dotenv import load_dotenv
import re


load_dotenv()


# model = ChatOpenAI(
#     model="gpt-4o-mini", 
#     temperature=0.7,
# )

model = ChatOllama(model="ministral-3",temperature=0.7)


class BlogState(TypedDict):
    title: str
    outline: str
    blog: str
    evaluationScore: int


def createOutline(state: BlogState) -> BlogState:
    blogtitle = state["title"]
    prompt = f"Create a brief outline (5 bullet points max) for a blog on: {blogtitle}"
    state["outline"] = model.invoke(prompt).content
    return state


def createBlog(state: BlogState) -> BlogState:
    outline = state["outline"]
    topic = state["title"]
    prompt = f"Write a blog post (200-250 words) based on this outline:\n{outline}\n\nBlog title: {topic}"
    state["blog"] = model.invoke(prompt).content
    return state


def evaluateBlog(state: BlogState) -> BlogState:
    prompt = f"Rate this blog out of 10. Respond with ONLY a single digit number (1-10), nothing else.\n\nOutline: {state['outline']}\n\nBlog: {state['blog']}"
    score_text = model.invoke(prompt).content.strip()
    
    # Extract first digit found
    numbers = re.findall(r'\d', score_text)
    score = int(numbers[0]) if numbers else 5
    state["evaluationScore"] = score
    
    return state


# Build graph
graph = StateGraph(BlogState)

graph.add_node("createOutline", createOutline)
graph.add_node("createBlog", createBlog)
graph.add_node("evaluateBlog", evaluateBlog)

graph.add_edge(START, "createOutline")
graph.add_edge("createOutline", "createBlog")
graph.add_edge("createBlog", "evaluateBlog")
graph.add_edge("evaluateBlog", END)

workflow = graph.compile()

# Run
initial_state = {"title": "bioinformatics"}
result = workflow.invoke(initial_state)

print(f"\n{'='*50}")
print(f"Title: {result['title']}")
print(f"{'='*50}")
print(f"\nOutline:\n{result['outline']}")
print(f"\n{'-'*50}")
print(f"Blog:\n{result['blog']}")
print(f"\n{'-'*50}")
print(f"Evaluation Score: {result['evaluationScore']}/10")
print(f"{'='*50}")