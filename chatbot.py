from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import chainlit as cl

# Load the pre-processed vector store from disk with the same embedding function
embedding_function = OllamaEmbeddings(model="mxbai-embed-large", num_gpu=0)
docsearch = Chroma(persist_directory="rag_files", embedding_function=embedding_function)

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
    msg = cl.Message(content="Your Chatbot is ready")
    await msg.send()

    # Store the chain in user session
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")
    # Callbacks happen asynchronously/parallel 
    cb = cl.AsyncLangchainCallbackHandler()
    
    # Call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
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
    await cl.Message(content=answer, elements=text_elements).send()
