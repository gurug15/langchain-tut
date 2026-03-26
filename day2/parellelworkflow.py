from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_ollama  import ChatOllama
from typing import TypedDict,Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import re
import operator



model = ChatOllama(model="ministral-3",temperature=0.7)

class EvaluationSchema(BaseModel):
     feedback: str = Field(description="detailed Feedback from the essay")
     score: int = Field(description="Score out of 10", gt=0 , lt=10)


structured_Model = model.with_structured_output(EvaluationSchema)




# prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {essay}'


# print(structured_Model.invoke(prompt))


class essayState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_score: Annotated[list[int], operator.add]
    avg_score: float



graph = StateGraph(essayState)


def eval_language(state:essayState):
    prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    result:EvaluationSchema = structured_Model.invoke(prompt)

    return {"language_feedback":result.feedback, "individual_score":[result.score]} 


def eval_analysis(state:essayState):
    prompt = f'Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    result:EvaluationSchema = structured_Model.invoke(prompt)

    return {"analysis_feedback":result.feedback, "individual_score":[result.score]}
     
def eval_clarity(state:essayState):
    prompt = f'Evaluate the clarity  of thought for the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    result:EvaluationSchema = structured_Model.invoke(prompt)

    return {"clarity_feedback":result.feedback, "individual_score":[result.score]}

def final_eval(state:essayState):
     # summary feedback
    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state["language_feedback"]} \n depth of analysis feedback - {state["analysis_feedback"]} \n clarity of thought feedback - {state["clarity_feedback"]}'
    overall_feedback = model.invoke(prompt).content

    # avg calculate
    avg_score = sum(state['individual_score'])/len(state['individual_score'])

    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}


graph.add_node("eval_language",eval_language)
graph.add_node("eval_analysis",eval_analysis)
graph.add_node("eval_clarity",eval_clarity)
graph.add_node("final_eval",final_eval)


graph.add_edge(START,"eval_language")
graph.add_edge(START, "eval_analysis")
graph.add_edge(START, "eval_clarity")
graph.add_edge("eval_language", "final_eval")
graph.add_edge("eval_analysis", "final_eval")
graph.add_edge("eval_clarity", "final_eval")

graph.add_edge("final_eval",END)

essay = """India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.

One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India’s doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.

To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

India’s demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India’s AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society."""



workflow = graph.compile()
initial_state = {'essay': essay}
print(workflow.invoke(initial_state))