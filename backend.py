from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import InMemorySaver



load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

model = ChatOpenAI(
    model="GPT-4.1",                     # base Sonar model
    base_url="https://api.perplexity.ai",
    api_key=PERPLEXITY_API_KEY,
    temperature=0.7
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
checkPoint = InMemorySaver()

def chat_node(state: ChatState):
    messages = state['messages']

    # Stronger system instruction
    system_message = {
        "role": "system",
        "content": (
            "You are a warm, friendly, *purely conversational* chatbot. "
            "You should NEVER explain what words mean unless explicitly asked. "
            "If a user greets you (e.g., 'hi', 'hello'), simply greet them back warmly. "
            "If the user gives their name, address them by name in future replies. "
            "Keep all responses short and natural, like a human chat friend."
        )
    }

    formatted_messages = [system_message]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_messages.append({"role": "assistant", "content": msg.content})
        else:
            formatted_messages.append(msg)

    response = model.invoke(formatted_messages)
    return {"messages": [response]}

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

checkPointer= InMemorySaver()


chatbot = graph.compile(checkpointer=checkPointer)

