# workflow.py

from dataclasses import dataclass, field
from typing import List, Any
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="sk-proj-pqdJir0wj64cUU5suUeOoPs9V_7sTEMRX2WJr521d5vnWA0SQJK0Eqlv92kqyvteXSkSLl-X4sT3BlbkFJOGd4o2FwQcn_7rIQdeMOfQt2qHYU9EaWPMFP-KMcDC2cN3uUV9guAHv5AIlOeCtyV2sJDPIk0A")

@dataclass
class WorkflowState:
    query: str
    tasks: List[str] = field(default_factory=list)
    current_task: int = 0
    results: List[Any] = field(default_factory=list)

def plan_agent_node(state: WorkflowState):
    prompt = f"""
    You are PlanAgent.
    Split the following user query into 3 clear sub-tasks.

    Query: {state.query}

    Format: 
    1. Sub-task 1
    2. Sub-task 2
    3. Sub-task 3
    """
    response = llm.invoke(prompt)
    sub_tasks = [
        line.strip().split(". ", 1)[-1]
        for line in response.content.strip().split("\n")
        if line.strip()
    ]
    state.tasks = sub_tasks
    state.current_task = 0
    state.results = []
    return state

def tool_agent_node(state: WorkflowState):
    task = state.tasks[state.current_task]
    prompt = f"You are ToolAgent. Solve the following sub-task:\n\nTask: {task}"
    result = llm.invoke(prompt).content.strip()
    state.results.append({"task": task, "result": result})
    state.current_task += 1
    return state

def reflection_node(state: WorkflowState):
    latest_result = state.results[-1]["result"]
    prompt = f"""
    You are ReflectionAgent.
    Evaluate the following result. Is it complete and correct? 
    Answer only 'YES' or 'NO' — if NO, suggest an improvement.

    Result: {latest_result}
    """
    reflection = llm.invoke(prompt).content.strip()
    if "YES" not in reflection.upper():
        state.tasks.append(f"Improve previous result: {latest_result}")
    return state

# Build LangGraph
graph = StateGraph(WorkflowState)
graph.add_node("PlanAgent", plan_agent_node)
graph.add_node("ToolAgent", tool_agent_node)
graph.add_node("Reflection", reflection_node)

graph.set_entry_point("PlanAgent")
graph.add_edge("PlanAgent", "ToolAgent")
graph.add_edge("ToolAgent", "Reflection")
graph.add_conditional_edges(
    "Reflection",
    lambda state: END if state.current_task >= len(state.tasks) else "ToolAgent"
)

app = graph.compile()

def run_langgraph_workflow(query: str):
    initial_state = WorkflowState(query=query)
    final_state = app.invoke(initial_state)
    return final_state
