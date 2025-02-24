import operator
from typing import Annotated, Any
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from pprint import pprint
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class State(BaseModel):
    query: str
    messages: Annotated[list[BaseMessage], operator.add] = Field(default=[])


def add_message(state: State) -> dict[str, Any]:
    additional_messages = []
    if not state.messages:
        additional_messages.append(
            SystemMessage(content="あなたは最小限の応答をする対話エージェントです。")
        )

    additional_messages.append(HumanMessage(content=state.query))
    return {"messages": additional_messages}


def llm_response(state: State) -> dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    ai_message = llm.invoke(state.messages)
    return {"messages": [ai_message]}


def print_checkpoint_dump(checkpointer: BaseCheckpointSaver, config: RunnableConfig):
    checkpoint_tuple = checkpointer.get_tuple(config)
    print("チェックポイントデータ:")
    pprint(checkpoint_tuple.checkpoint)
    print("\nメタデータ:")
    pprint(checkpoint_tuple.metadata)


graph = StateGraph(State)
graph.add_node("add_message", add_message)
graph.add_node("llm_response", llm_response)
graph.set_entry_point("add_message")
graph.add_edge("add_message", "llm_response")
graph.add_edge("llm_response", END)

checkpointer = MemorySaver()

compiled_graph = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "example-1"}}

user_query = State(query="私の好きなものはずんだ餅です。覚えておいてね。")

first_response = compiled_graph.invoke(user_query, config)
print("==========start first response================")
print(first_response)
print("==============================================")

for checkpoint in checkpointer.list(config):
    print(checkpoint)

print("==============================================")
print_checkpoint_dump(checkpointer, config)
print("==========end first response==================")

user_query = State(query="私の好物は何か覚えてる？")
second_response = compiled_graph.invoke(user_query, config)
print("==========start second response================")
print(second_response)
print("==============================================")

for checkpoint in checkpointer.list(config):
    print(checkpoint)

print("==============================================")
print_checkpoint_dump(checkpointer, config)
print("==========end second response==================")
