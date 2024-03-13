import streamlit as st 
from streamlit_chat import message
import tempfile
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.llms import VLLM
from transformers import pipeline
from transformers import AutoTokenizer
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import RetrievalQAWithSourcesChain
import csv
from typing import Dict, List, Optional
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document



#Loading the model
@st.cache_resource()
def load_llm():
    # Load the locally downloaded model here
    # bnb_config = transformers.BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type='nf4',
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_compute_dtype='bfloat16'
    #     )
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = transformers.AutoModelForCausalLM.from_pretrained(
  model_name,
  trust_remote_code=True,
  #quantization_config=bnb_config,
  device_map = 'auto'
)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer,max_new_tokens = 1024,top_k = 10,top_p = 0.95,temperature = 0.01)
    llm=HuggingFacePipeline(pipeline=pipe)
    return llm



class MetaDataCSVLoader(BaseLoader):
    """Loads a CSV file into a list of documents.

    Each document represents one row of the CSV file. Every row is converted into a
    key/value pair and outputted to a new line in the document's page_content.

    The source for each document loaded from csv is set to the value of the
    `file_path` argument for all doucments by default.
    You can override this by setting the `source_column` argument to the
    name of a column in the CSV file.
    The source of each document will then be set to the value of the column
    with the name specified in `source_column`.

    Output Example:
        .. code-block:: txt

            column1: value1
            column2: value2
            column3: value3
    """

    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,   # < ADDED
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = None,
    ):
        self.file_path = file_path
        self.source_column = source_column
        self.encoding = encoding
        self.csv_args = csv_args or {}
        self.metadata_columns = metadata_columns        # < ADDED

    def load(self) -> List[Document]:
        """Load data into document objects."""  

        docs = []
        with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
            csv_reader = csv.DictReader(csvfile, **self.csv_args)  # type: ignore
            for i, row in enumerate(csv_reader):
                content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in row.items())
                try:
                    source = (
                        row[self.source_column]
                        if self.source_column is not None
                        else self.file_path
                    )
                except KeyError:
                    raise ValueError(
                        f"Source column '{self.source_column}' not found in CSV file."
                    )
                metadata = {"source": source, "row": i}
                # ADDED TO SAVE METADATA
                if self.metadata_columns:
                    for k, v in row.items():
                        if k in self.metadata_columns:
                            metadata[k] = v
                # END OF ADDED CODE
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)

        return docs

#Streamlit App
logo_path = "./Philips_logo.png"
st.sidebar.image(logo_path)
st.title("Chat with OHC Lessons Learned Database")
#st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)
with st.spinner('Loading Model Please wait...'):
    llm = load_llm()
    st.success('Model Loaded Successfully!')

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    #use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

    model_name='sentence-transformers/all-mpnet-base-v2'
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)

    # Load data and set embeddings 
    loader = MetaDataCSVLoader(file_path="./4_data/4_data_merged_without_null.csv",metadata_columns=['Project', 'Project Phase',
        "What Happened? And why do we care?", "Why did it happen?", "Do's",
        "Type", "Tip group", "Reviewed in PIR"]) #<= modified 
    data = loader.load()

    #data = loader.load()
    vectordb = Chroma.from_documents(data, embedding_function)

    metadata_field_info=[    
        AttributeInfo(
            name="Project",
            description="Name of the Project. NOTE: Only use the 'eq' operator if a specific Project Name is mentioned. If a Project is not mentioned, include all", 
            type="string", 
        ),
        ]
      
    document_content_description = "Description of various Projects"
    retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    #enable_limit=True,
    verbose=True
    )
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm, 
                                                        chain_type="stuff", 
                                                        retriever=retriever)


    def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]
        
    if 'history' not in st.session_state:
            st.session_state['history'] = []

    if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me anything about " + "EarlyBird Project" + " 🤗"]

    if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]
            
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
            with st.form(key='my_form', clear_on_submit=True):
                
                user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")


