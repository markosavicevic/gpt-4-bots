import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Load environment variables
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENV"]

# Initialize retrieval components
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

# index_name = "pod-gpt"
index_name = "scientific-principles-gpt"
index = pinecone.Index(index_name)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vectordb = Pinecone(
    index=index,
    embedding_function=embeddings.embed_query,
    text_key="text"
)

# Initialize chat model
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0.7,
    model_name='gpt-4'
)

# Initialize QA retrieval object
retriever = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever()
)

# Set up conversational agent
tool_desc = """Use this tool to answer user questions using Scientific Principles of Hypertrophy Training.
If the user states 'ask Mike' use this tool to get the answer. This tool can also be used for follow up
questions from the user."""

tools = [Tool(
    func=retriever.run,
    description=tool_desc,
    name='Scientific Priciples',
)]

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True
)

conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=2,
    early_stopping_method="generate",
    memory=memory,
)

# Add this code snippet to set up the conversational agent prompt
sys_msg = """You are Mike Israetel. You answers the user's questions and complete the user's tasks.\n"""

prompt = conversational_agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)
conversational_agent.agent.llm_chain.prompt = prompt

# Test the conversational agent with a sample question
# question = "ask lex about the future of ai"
# response = conversational_agent(question)
# print(response)

# Test the conversational agent with a sample question
# question = "ask Lex: Does Lex think Bob Lazar is lying?"
# response = conversational_agent(question)
# print(response)

# Add a loop to continuously chat with the bot
# print("Type 'quit' to exit.")
# while True:
#    user_input = input("User: ")
#    if user_input.lower() == "quit":
#        break

#    response = conversational_agent(user_input)
#    print("Bot:", response['output'])