Agentic RAG

This project is a multimodal retrieval-augmented generation (RAG) system that integrates various AI tools, including HuggingFace embeddings, Groq models, LangChain, and Chroma, for advanced document retrieval and question answering. It utilizes structured workflows to perform tasks like document retrieval, question refinement, and answer generation based on retrieved content.

Features

Document Retrieval: Automatically retrieves documents from specified URLs and splits them for further processing.

Grading: Assess the relevance of retrieved documents for a given user query and decide whether to generate an answer or rewrite the question.

Question Rewriting: Refines user queries to improve the semantic alignment with the documents.

Answer Generation: Uses RAG to generate answers based on the retrieved and relevant documents.

Setup

Prerequisites

Python 3.7+

API Keys: You need to set up .env with the following API keys:

GOOGLE_API_KEY: Google API key for access to relevant services.

TAVILY_API_KEY: Tavily API key for enhanced data processing.

GROQ_API_KEY: API key for using Groq models.

LANGCHAIN_API_KEY: LangChain API key for orchestrating AI tools.

Add your API keys to a .env file like below:

ini
Copy
Edit
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
Installation
Clone the repository:


Google API: Used for any Google-related data extraction or processing.

Tavily API: For accessing additional content or functionalities.

Groq API: For running Groq-based LLMs, such as Gemma2-9b-It.

LangChain API: Used for chaining different tools and models for RAG workflows.

Workflow


1. Document Loading
   
Documents are retrieved from the following URLs:

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

These URLs are loaded using LangChain's WebBaseLoader.

2. Document Splitting
The documents are split into smaller chunks using LangChain's RecursiveCharacterTextSplitter to ensure they are manageable for the model:


The system is designed to process documents, evaluate relevance, and generate natural language responses in a dynamic and efficient manner.

Contributing
Feel free to fork this repository, submit issues, or contribute to the code. For major changes, please open an issue first to discuss what you would like to change.

License
This project is licensed under the MIT License - see the LICENSE file for details.

