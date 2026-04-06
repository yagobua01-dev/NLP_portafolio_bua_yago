import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# UI Configuration
st.set_page_config(page_title="Local PDF Assistant", layout="centered")
st.title("Local PDF Assistant")
st.write("Upload a PDF document and ask questions about it using a local LLM.")

# System Components Configuration
# We use 'mistral' as our local LLM and 'nomic-embed-text' for embeddings.
# Both must be downloaded in Ollama prior to running this app.
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"

@st.cache_resource
def setup_pipeline(file_path):
    """
    This function handles the preprocessing, embeddings, and vector database creation.
    It is cached to avoid reprocessing the PDF on every user interaction.
    """
    # Step 1: Load the document
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Step 2: Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Step 3: Create embeddings and store them in a local vector database
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Step 4: Set up the LLM and the prompt template
    llm = Ollama(model=LLM_MODEL)
    
    system_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, say that you don't know. "
        "Keep the answer concise.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Step 5: Create the RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# User Interaction
uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.success("PDF uploaded successfully! Processing document...")
    
    try:
        # Initialize the NLP pipeline
        chain = setup_pipeline(tmp_file_path)
        st.success("Document processed. You can now ask questions!")
        
        # Get user question
        user_question = st.text_input("What do you want to know about the document?")
        
        if user_question:
            with st.spinner("Thinking..."):
                # Run the chain (Retrieval + Generation)
                response = chain.invoke({"input": user_question})
                
                # Display the results
                st.subheader("Answer:")
                st.write(response["answer"])
                
                # Added value: Show the source documents used for the answer
                with st.expander("Show source chunks used"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"**Chunk {i+1}:** {doc.page_content}")
                        
    finally:
        # Clean up the temporary file
        os.remove(tmp_file_path)

else:
    st.info("Please upload a PDF file to begin.")