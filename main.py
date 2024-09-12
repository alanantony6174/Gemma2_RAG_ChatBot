import os.path
import pickle
from typing import Dict, List, Optional

import chainlit.data as cl_data
from chainlit.socket import persist_user_session
from chainlit.step import StepDict
from literalai.helper import utc_now


import chainlit as cl

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


now = utc_now()

thread_history = []  # type: List[cl_data.ThreadDict]
deleted_thread_ids = []  # type: List[str]

THREAD_HISTORY_PICKLE_PATH = os.getenv("THREAD_HISTORY_PICKLE_PATH")
if THREAD_HISTORY_PICKLE_PATH and os.path.exists(THREAD_HISTORY_PICKLE_PATH):
    with open(THREAD_HISTORY_PICKLE_PATH, "rb") as f:
        thread_history = pickle.load(f)

async def save_thread_history():
    if THREAD_HISTORY_PICKLE_PATH:
        # Force saving of thread history for reload when server restarts
        await persist_user_session(
            cl.context.session.thread_id, cl.context.session.to_persistable()
        )

        with open(THREAD_HISTORY_PICKLE_PATH, "wb") as out_file:
            pickle.dump(thread_history, out_file)

class TestDataLayer(cl_data.BaseDataLayer):
    async def get_user(self, identifier: str):
        return cl.PersistedUser(id="test", createdAt=now, identifier=identifier)

    async def create_user(self, user: cl.User):
        return cl.PersistedUser(id="test", createdAt=now, identifier=user.identifier)

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        thread = next((t for t in thread_history if t["id"] == thread_id), None)
        if thread:
            if name:
                thread["name"] = name
            if metadata:
                thread["metadata"] = metadata
            if tags:
                thread["tags"] = tags
        else:
            thread_history.append(
                {
                    "id": thread_id,
                    "name": name,
                    "metadata": metadata,
                    "tags": tags,
                    "createdAt": utc_now(),
                    "userId": user_id,
                    "userIdentifier": "admin",
                    "steps": [],
                }
            )

    @cl_data.queue_until_user_message()
    async def create_step(self, step_dict: StepDict):
        cl.user_session.set(
            "create_step_counter", cl.user_session.get("create_step_counter") + 1
        )

        thread = next(
            (t for t in thread_history if t["id"] == step_dict.get("threadId")), None
        )
        if thread:
            thread["steps"].append(step_dict)

    async def get_thread_author(self, thread_id: str):
        return "admin"

    async def list_threads(
        self, pagination: cl_data.Pagination, filters: cl_data.ThreadFilter
    ) -> cl_data.PaginatedResponse[cl_data.ThreadDict]:
        return cl_data.PaginatedResponse(
            data=[t for t in thread_history if t["id"] not in deleted_thread_ids],
            pageInfo=cl_data.PageInfo(
                hasNextPage=False, startCursor=None, endCursor=None
            ),
        )

    async def get_thread(self, thread_id: str):
        thread = next((t for t in thread_history if t["id"] == thread_id), None)
        if not thread:
            return None
        thread["steps"] = sorted(thread["steps"], key=lambda x: x["createdAt"])
        return thread

    async def delete_thread(self, thread_id: str):
        deleted_thread_ids.append(thread_id)

cl_data._data_layer = TestDataLayer()

# RAG System Configuration
# embedding_function = OllamaEmbeddings(model="mxbai-embed-large", num_gpu=0)
embedding_function = OllamaEmbeddings(model="mxbai-embed-large")
docsearch = Chroma(persist_directory="ocr_rag_files", embedding_function=embedding_function)

async def send_count():
    create_step_counter = cl.user_session.get("create_step_counter")
    #await cl.Message(f"Create step counter: {create_step_counter}").send()

@cl.on_chat_start
async def on_chat_start():

    # Initialize message history for conversation
    message_history = ChatMessageHistory()
    
    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model="gemma2:2b"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg = cl.Message(content="Your Digitaiken's Chatbot is ready")
    await msg.send()

    # Store the chain in user session
    cl.user_session.set("chain", chain)
    cl.user_session.set("create_step_counter", 0)

    await send_count()

@cl.on_message
async def handle_message(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")

    # Call the chain with user's message content
    res = await chain.ainvoke(message.content)
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []  # Initialize list to store text elements
    
    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        
        # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    # Return results
    # await cl.Message(content=answer, elements=text_elements).send()
    await cl.Message(content=answer).send()
    
    await send_count()
    # async with cl.Step(type="tool", name="thinking") as step:
    #     step.output = "Thinking..."
    # await cl.Message("Ok!").send()
    await save_thread_history()

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    username_stored = os.getenv("CHAINLIT_USERNAME")
    password_stored = os.getenv("CHAINLIT_PASSWORD")

    if username_stored is None or password_stored is None:
        raise ValueError(
            "Username or password not set. Please set CHAINLIT_USERNAME and "
            "CHAINLIT_PASSWORD environment variables."
        )

    if (username, password) == (username_stored, password_stored):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.on_chat_resume
async def on_chat_resume(thread: cl_data.ThreadDict):
    # Reinitialize the chain object
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model="gemma2:2b"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    
    # Store the chain in user session
    cl.user_session.set("chain", chain)

    # Send a welcome back message without metadata or tags
    await cl.Message(f"Welcome back to {thread['name']}").send()

 