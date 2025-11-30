import os
import sys
from typing import Annotated, Literal
from functools import partial

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from tools import get_inflation_factor, get_housing_statistics


class State(BaseModel):
    messages: Annotated[list, add_messages]

def agent(state: State, model):
    return {"messages": [model.invoke(state.messages)]}

def getLLMModel(model_name, tools):
    llm = ChatOpenAI(model=model_name)
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools

def agentBuilder(tools, model):
    builder = StateGraph(State)
    builder.add_node("agent", partial(agent, model=model))
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("agent", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph

def main(model_name, tools):
    """Configuration for the conversation thread"""
    model = getLLMModel(model_name=model_name, tools=tools)
    config = {"configurable": {"thread_id": "1"}}

    user_input_1 = "What is the average house price?"
    print(f"User: {user_input_1}")
    graph = agentBuilder(tools=tools, model=model)
    events = graph.stream(
        {"messages": [HumanMessage(content=user_input_1)]},
        config,
        stream_mode="values"
    )
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    user_input_2 = "what was the inflation factor?"
    print(f"User: {user_input_2}")
    
    events = graph.stream(
        {"messages": [HumanMessage(content=user_input_2)]},
        config,
        stream_mode="values"
    )
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

if __name__ == "__main__":
    tools = [get_inflation_factor, get_housing_statistics]
    model_name = "gpt-5-nano-2025-08-07"  
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
    
    main(model_name=model_name, tools=tools)
