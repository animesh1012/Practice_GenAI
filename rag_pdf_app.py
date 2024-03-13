import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from transformers import AutoTokenizer
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from langchain_community.llms import VLLM
import torch

torch.cuda.set_device(torch.device('cuda:0'))

#Extracting Text of Pdf
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

#Creating chunks from text
def get_text_chunks(text):
    #Creating structure of chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

#Vectorizing Chunks
def get_vector_store(text_chunks):
    model_name='sentence-transformers/all-mpnet-base-v2'
    embedding_fn = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_texts(text_chunks,embedding = embedding_fn)
    vector_store.save_local("./faiss_index")

#Loading the model
def load_llm(model_name):
    # Load the locally downloaded model here
    bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype='bfloat16'
        )
    model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map = 'auto'
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline('text-generation', model=model,max_new_tokens = 512,tokenizer=tokenizer,top_k = 10,top_p = 0.95,temperature = 0.3)
    llm=HuggingFacePipeline(pipeline=pipe)
#     llm = VLLM(
#     model=model_name,
#     #model = model_name,
#     trust_remote_code=True,  # mandatory for hf models
#     max_new_tokens=128,
#     top_k=10,
#     top_p=0.95,
#     temperature=0.8,
#     gpu_memory_utilization=0.80,
# )
    return llm

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    
    Answer:
    """
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    #model_name = "Llama-2-13b-chat-hf-quantized"
    model = load_llm(model_name)
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain = LLMChain(llm = model,prompt=prompt)
    
    return chain

def find_answer(user_question):
    model_name='sentence-transformers/all-mpnet-base-v2'
    embedding_fn = HuggingFaceEmbeddings(model_name=model_name)
    new_db = FAISS.load_local("faiss_index",embedding_fn)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    # response = chain(
    #     {'input_documents':docs,'question':user_question},return_only_outputs=True
    # )
    response = chain.run(question=user_question, context=docs)
    print(response)
    st.write("Reply: ",response)


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using LLM💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        find_answer(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()