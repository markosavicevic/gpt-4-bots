#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ### Load your data

# In[2]:


loader = UnstructuredPDFLoader("Marcus Aurelius, Gregory Hays - Meditations_ A New Translation-Modern Library (2003).pdf")
# loader = OnlinePDFLoader("https://www.ogilvy.com/filedownload?f_path=L3NpdGVzL2cvZmlsZXMvZGhwc2p6MTA2L2ZpbGVzL3BkZmRvY3VtZW50cy9PZ2lsdnklMjAtJTIwVGhlJTIwU2hpZnQlMjBmcm9tJTIwSW1hZ2UlMjB0byUyMEltcGFjdC5wZGY%3D&force_download=1")


# In[3]:


data = loader.load()


# In[4]:


print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')


# ### Chunk your data up into smaller documents

# In[5]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)


# In[6]:


print (f'Now you have {len(texts)} documents')


# ### Create embeddings of your documents to get ready for semantic search

# In[9]:


from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone


# In[24]:


OPENAI_API_KEY = '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'
PINECONE_API_KEY = '7752d52a-1fcb-4853-9e51-b9c21d213bb4'
PINECONE_API_ENV = 'us-east1-gcp'


# In[11]:


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# In[12]:


# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "meditations-gpt"


# In[13]:


docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)


# In[14]:


query = "What is hypertrophy?"
docs = docsearch.similarity_search(query, include_metadata=True)


# ### Query those docs to get your answer back

# In[16]:


from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


# In[17]:


llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")


# In[22]:


query = "Why is hypertrophy good?"
docs = docsearch.similarity_search(query, include_metadata=True)


# In[23]:


chain.run(input_documents=docs, question=query)


# In[ ]:




