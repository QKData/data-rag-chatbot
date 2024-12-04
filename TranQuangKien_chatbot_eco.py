from operator import itemgetter
from time import sleep
import os

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeVectorStore
from llama_parse import LlamaParse
import nest_asyncio
import tempfile
from langchain_core.documents import Document


nest_asyncio.apply()
os.environ["LLAMA_CLOUD_API_KEY"] = ""

# Parse PDF file
def parse_pdf_file(pdf_file):
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_pdf_path = temp_pdf.name

    # Use LlamaParse to parse the PDF
    llama_parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown"
    )
    
    documents = llama_parser.load_data(temp_pdf_path)

    # Combine all document texts
    full_text = "\n\n".join(doc.text for doc in documents)
    
    # Clean up the temporary file
    os.unlink(temp_pdf_path)

    # Split the content into chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400, length_function=len)
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1")])
    chunks = text_splitter.split_text(full_text)
    chunks = [chunk.page_content for chunk in chunks]
    
    # Create Document objects with unique metadata for each chunk
    documents = []
    for i, chunk in enumerate(chunks):
        metadata = {
            'pdf_file': pdf_file.name,
            'chunk_id': i,
            'total_chunks': len(chunks)
        }
        documents.append(Document(page_content=chunk, metadata=metadata))
    
    # Add documents to the vector store
    st.session_state.vector_store.add_documents(documents)
    return f'Added {len(chunks)} chunks to the index.'

# Initialize session state for global variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    # OpenAI API key and Pinecone API key
    st.session_state.openai_api_key = ""
    st.session_state.pinecone_api_key = ""

    os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
    os.environ['PINECONE_API_KEY'] = st.session_state.pinecone_api_key

    # Initialize Pinecone
    st.session_state.pc = Pinecone(api_key=st.session_state.pinecone_api_key)
    st.session_state.index_name = "md-eco-chatbot"

    # Check if index exists, if not create it
    index_list = st.session_state.pc.list_indexes()
    if st.session_state.index_name not in [index.name for index in index_list]:
        st.session_state.pc.create_index(
            name=st.session_state.index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

        # Wait for the index to be created
        while not st.session_state.pc.describe_index(st.session_state.index_name).status['ready']:
            st.write("Waiting for the index to be created...")
            sleep(1)

    # Get the index
    st.session_state.index = st.session_state.pc.Index(st.session_state.index_name)

    # Initialize the vector store
    st.session_state.embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
    st.session_state.text_field = 'text'
    st.session_state.vector_store = PineconeVectorStore(
        index=st.session_state.index,
        embedding=st.session_state.embedding_model,
        text_key=st.session_state.text_field
    )

    # Initialize the chat model
    st.session_state.chatbot = ChatOpenAI(model='gpt-4o-mini')

    # Set up memory
    st.session_state.memory = ConversationTokenBufferMemory(
        memory_key="history",
        return_messages=True,
        max_token_limit=2048,
        llm=st.session_state.chatbot
    )

    # Set up the chat prompt template
    st.session_state.template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Relevant pieces of previous conversation:
    {history}

    (You do not need to use these pieces of information if not relevant)

    The context of the current question is:
    {context}

    Current conversation:
    Question: {question}
    AI: """
    st.session_state.prompt = ChatPromptTemplate.from_template(st.session_state.template)

    # Set up the retriever
    st.session_state.retriever = st.session_state.vector_store.as_retriever(search_type="similarity",  # You can also try "mmr" for maximum marginal relevance
                                                                            search_kwargs={"k": 5})

    # Set up the document loader
    st.session_state.setup_and_retrieval = RunnableParallel(
        {
            'context': st.session_state.retriever,
            'question': RunnablePassthrough(),
            'history': RunnableLambda(st.session_state.memory.load_memory_variables) | itemgetter('history'),
        }
    )

    # Output parser
    st.session_state.output_parser = StrOutputParser()

    # Set up the chain
    st.session_state.chain = st.session_state.setup_and_retrieval | st.session_state.prompt | st.session_state.chatbot | st.session_state.output_parser

    st.session_state.initialized = True

# Streamlit app
st.title("Economic Analysis Chatbot")

# Add a section for uploading PDF files, do not show the file content
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_file")
if pdf_file:
    st.write("PDF file uploaded successfully!")
    if st.button("Parse PDF and Add to Knowledge Base"):
        if pdf_file:
            pdf_text = parse_pdf_file(pdf_file)
            st.write(pdf_text)
        else:
            st.write("Please upload a PDF file first.")

# Display the chat history from memory
chat_history = st.session_state.memory.buffer_as_messages
for message in chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Move the input field to the bottom
user_question = st.chat_input("Ask a question:")

# React to the user input
if user_question:
    # Display the user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_question)

    # Run the chatbot
    response = st.session_state.chain.invoke(user_question)

    # Store the response in memory
    st.session_state.memory.save_context(
    {
        'input': user_question,
    },
    {
        'output': response,
    })

    # Rerun the chatbot chain with the updated memory
    st.rerun()
