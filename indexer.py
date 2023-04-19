from typing import List, Optional
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import openai
from tqdm.auto import tqdm
import os
from models import Record, Metadata
import json

tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text: str) -> int:
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

class Chunker:
    def __init__(self, chunk_size: Optional[int] = 400, chunk_overlap: Optional[int] = 20):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=tiktoken_len,
            separators=['\n\n', '.\n', '\n', '.', '?', '!', ' ', '']
        )

    def __call__(self, title: str, text: str, source: str, part: Optional[str] = None, chapter: Optional[int] = None) -> List[Record]:
        text_chunks = self.text_splitter.split_text(text)
        return [
            Record(
                id=f'{title}-{part}-{chapter}-{i}',
                text=text,
                metadata=Metadata(
                    title=title,
                    part=part,
                    chapter=chapter,
                    source=source,
                    chunk=i
                )
            )
            for i, text in enumerate(text_chunks)
        ]

class Indexer:
    dimension_map = {
        'text-embedding-ada-002': 1536
    }
    def __init__(
        self, openai_api_key: Optional[str], pinecone_api_key: Optional[str],
        pinecone_environment: Optional[str], index_name: Optional[str] = "book-gpt",
        embedding_model_name: Optional[str] = "text-embedding-ada-002",
        chunk_size: Optional[int] = 400, chunk_overlap: Optional[int] = 20
    ):
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        self.pinecone_api_key = pinecone_api_key or os.environ.get('PINECONE_API_KEY')
        self.pinecone_environment = pinecone_environment or os.environ.get('PINECONE_ENVIRONMENT')
        if self.openai_api_key is None:
            raise ValueError('openai_api_key not specified')
        if self.pinecone_api_key is None:
            raise ValueError('pinecone_api_key not specified')
        if self.pinecone_environment is None:
            raise ValueError('pinecone_environment not specified')

        self.chunker = Chunker(chunk_size, chunk_overlap)
        self.embedding_model_name = embedding_model_name
        self.metadata_config = {'indexed': list(Metadata.schema()['properties'].keys())}

        pinecone.init(
            api_key=pinecone_api_key, environment=pinecone_environment
        )

        openai.api_key = openai_api_key

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name, dimension=self.dimension_map[embedding_model_name],
                metadata_config=self.metadata_config
            )

        self.index = pinecone.Index(index_name)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        res = openai.Embedding.create(
            input=texts,
            engine=self.embedding_model_name
        )
        return [result["embedding"] for result in res["data"]]

    def _index(self, records: List[Record]) -> None:
        ids = [record.id for record in records]
        texts = [record.text for record in records]
        metadatas = [json.dumps(dict(record.metadata)) for record in records]  # Convert metadata to a string

        for i, metadata in enumerate(metadatas):
            metadata_dict = json.loads(metadata)  # Convert metadata string to a dictionary
            metadata_dict['text'] = texts[i]  # Update the dictionary with the new key-value pair
            metadatas[i] = json.dumps(metadata_dict)  # Convert the updated dictionary back to a string

        embeddings = self._embed(texts)
        self.index.upsert(vectors=list(zip(ids, embeddings)), metadata=metadatas)  # Pass metadata as strings
    
    def process_section(self, title: str, text: str, source: str, part: Optional[str], chapter: Optional[int], batch_size: Optional[int] = 100) -> None:
        chunks = self.chunker(title=title, text=text, source=source, part=part, chapter=chapter)
        for i in range(0, len(chunks), batch_size):
            i_end = min(i + batch_size, len(chunks))
            self._index(chunks[i:i_end])    

    def __call__(self, title: str, text: str, source: str, part: Optional[str], chapter: Optional[int], batch_size: Optional[int] = 100) -> None:
        self.process_section(title=title, text=text, source=source, part=part, chapter=chapter, batch_size=batch_size)

        # Deinitialize Pinecone after using it
        pinecone.deinitialize()