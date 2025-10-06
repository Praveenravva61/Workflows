import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from backend import chatbot
import uuid

#___________________________________________________________Utilities________________________________________________________________________________

def generate_thread_id():
    """Generate a unique conversation/thread ID."""
    return str(uuid.uuid4())


def add_thread(thread_id):
    """Add a new thread to the session if not already there."""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def reset_chat():
    """Reset chat by creating a new thread and clearing history."""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []


def load_conversation(thread_id):
    """Load messages for a specific thread."""
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


#__________________________________________________________Sidebar UI_______________________________________________________________________________________________

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

st.session_state.setdefault('chat_threads', [])

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            temp_messages.append({'role': role, 'content': msg.content})
        st.session_state['message_history'] = temp_messages


#___________________________________________________________Session setup__________________________________________________________________________________

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state['chat_threads'] = []

add_thread(st.session_state['thread_id'])


#___________________________________________________________Main Chat UI__________________________________________________________________________________

# Display previous messages
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# Input box for user
user_input = st.chat_input('Type your message here...')

if user_input:
    # Add user's message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # Stream AI response
    with st.chat_message('assistant'):
        def ai_only_stream():
            for msg_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(msg_chunk, AIMessage):
                    yield msg_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    # Save assistant’s response
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
