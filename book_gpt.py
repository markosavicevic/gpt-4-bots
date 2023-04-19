from models import Record
from indexer import Indexer
import os

def read_text_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def process_book(book_path: str, title: str, author: str, source: str):
    book_content = read_text_file(book_path)

    # Split the book content into sections or chapters
    
    example_sections = [
        {"part": "Foreword", "chapter": None, "text": "Foreword content..."},
        {"part": "Preface to the Boardroom Edition", "chapter": None, "text": "Preface content..."},
        {"part": "Introduction", "chapter": None, "text": "Introduction content..."},
        {"part": "Contents", "chapter": None, "text": "Contents content..."},
        {"part": "Part I - The Basic Strategy of Persuasion", "chapter": 1, "text": "Chapter 1 content..."},
        {"part": "Part I - The Basic Strategy of Persuasion", "chapter": 2, "text": "Chapter 2 content..."},
        {"part": "Part I - The Basic Strategy of Persuasion", "chapter": 3, "text": "Chapter 3 content..."},
        {"part": "Part I - The Basic Strategy of Persuasion", "chapter": 4, "text": "Chapter 4 content..."},
        {"part": "Part I - The Basic Strategy of Persuasion", "chapter": 5, "text": "Chapter 5 content..."},
        {"part": "Part II - The Seven Basic Techniques of Breakthrough Advertising", "chapter": 6, "text": "Chapter 6 content..."},
        {"part": "Part II - The Seven Basic Techniques of Breakthrough Advertising", "chapter": 7, "text": "Chapter 7 content..."},
        {"part": "Part II - The Seven Basic Techniques of Breakthrough Advertising", "chapter": 8, "text": "Chapter 8 content..."},
        {"part": "Part II - The Seven Basic Techniques of Breakthrough Advertising", "chapter": 9, "text": "Chapter 9 content..."},
        {"part": "Part II - The Seven Basic Techniques of Breakthrough Advertising", "chapter": 10, "text": "Chapter 10 content..."},
        {"part": "Part II - The Seven Basic Techniques of Breakthrough Advertising", "chapter": 11, "text": "Chapter 11 content..."},
        {"part": "Part II - The Seven Basic Techniques of Breakthrough Advertising", "chapter": 12, "text": "Chapter 12 content..."},
        {"part": "Part II - The Seven Basic Techniques of Breakthrough Advertising", "chapter": 13, "text": "Chapter 13 content..."},
        {"part": "Part II - The Seven Basic Techniques of Breakthrough Advertising", "chapter": 14, "text": "Chapter 14 content..."},
        {"part": "Epilogue: A Copy Writer’s Library", "chapter": None, "text": "Epilogue content..."},
    ]

    indexer = Indexer(
        openai_api_key="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",
        pinecone_api_key="7752d52a-1fcb-4853-9e51-b9c21d213bb4",
        pinecone_environment="us-east1-gcp",
        index_name="book-gpt"
    )

    for section in example_sections:
        indexer(title=title, text=section["text"], source=source, part=section["part"], chapter=section["chapter"])

if __name__ == "__main__":
    book_path = "Breakthrough Advertising - Eugene Schwartz.txt"  # Add the path to the book .txt file
    title = "Breakthrough Advertising"
    author = "Eugene Schwartz"  # Add the author's name
    source = ""  # Add the source URL, if any

    process_book(book_path, title, author, source)