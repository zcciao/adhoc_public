from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def create_summarization_chain() -> LLMChain:
    """Create a LangChain chain for summarizing news articles."""
    # Initialize the OpenAI model
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo-16k",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a news summarizer. Your task is to create concise, informative summaries of news articles."),
        ("human", "Please summarize the following news article in 3-5 bullet points, highlighting the key information:\n\n{text}")
    ])

    # Create the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def summarize_news(content: str, max_chunk_size: int = 10000) -> str:
    """
    Summarize a news article using LangChain and OpenAI.
    
    Args:
        content: The content of the news article to summarize
        max_chunk_size: Maximum size of each chunk for processing
        
    Returns:
        A summarized version of the news article
    """
    # Create the summarization chain
    chain = create_summarization_chain()
    
    # Split the content into chunks if it's too long
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(content)
    
    # Process each chunk and combine the summaries
    summaries = []
    for chunk in chunks:
        summary = chain.invoke({"text": chunk})
        summaries.append(summary["text"])
    
    # If we have multiple chunks, combine their summaries
    if len(summaries) > 1:
        combined_summary = "\n\n".join(summaries)
        final_summary = chain.invoke({"text": combined_summary})
        return final_summary["text"]
    
    return summaries[0]

def extract_key_points(content: str) -> List[str]:
    """
    Extract key points from a news article.
    
    Args:
        content: The content of the news article
        
    Returns:
        List of key points
    """
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo-16k",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a news analyst. Extract the 5 most important key points from the following news article."),
        ("human", "Extract the key points from this article:\n\n{text}")
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"text": content})
    
    # Split the response into individual points
    key_points = [point.strip() for point in result["text"].split("\n") if point.strip()]
    return key_points 