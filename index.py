import os
import streamlit as st

# Uncomment to specify your OpenAI API key here (local testing only, not in production!), or add corresponding environment variable (recommended)
os.environ['OPENAI_API_KEY'] = "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"


from llama_index import VectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain.llms.openai import OpenAI

# Define a simple Streamlit app
st.set_page_config(page_title="🤗💬 GRG AI Chat")
st.title("GRG Support AI")
# Initialize an empty list to store chat messages
chat_messages = []

# Hugging Face Credentials
with st.sidebar:
    st.title('🤗💬 GRG Support AI')
    if ('TenantID' in st.secrets) and ('PASS' in st.secrets):
        st.success('Login credentials already provided!', icon='✅')
        hf_email = st.secrets['TenantID']
        hf_pass = st.secrets['PASS']
    else:
        hf_email = st.text_input('Enter Tanent-Id:', type='password')
        hf_pass = st.text_input('Enter password:', type='password')
        if not (hf_email and hf_pass):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

        # Configure prompt parameters and initialise helper
        max_input_size = 4096
        num_output = 256
        max_chunk_overlap = 0.2
        prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

        # Load documents from the 'data' directory
        documents = SimpleDirectoryReader('data').load_data()
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        response = index.as_query_engine().query(prompt)
        # Convert the "Response" object to a string
        response_str = str(response)
        full_response += (response_str or "")
        message_placeholder.markdown(full_response + "")
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
    

