from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain_core.documents.base import Document
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from operator import itemgetter
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Auto-prompt-builder"

with open("openai-prompting.txt", "r") as f:
    text = f.read()

template = """
> <instructions>: {text} </instructions>

--------------------
--------------------
--------------------

Based on the instructions above, (which are delimited by XML tag), please assist me in writing an effective prompt template for my objective.

My objective is enclosed within triple backticks (```).
It is crucial for my career, so please do your best.

return your response in the following format:
prompt: ...

> ```{objective}```
"""

splitter = TokenTextSplitter(chunk_size=2000,chunk_overlap=100)
chunks = splitter.split_text(text)

docs = [Document(page_content=chunk) for chunk in chunks]

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
openai = ChatOpenAI(model="gpt-3.5-turbo-1106",temperature=0)
google = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0)

url = os.getenv("CLUSTER_URL")
api_key = os.getenv("CLUSTER_API_KEY")
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="my_documents")

retriever = qdrant.as_retriever(search_type="mmr",search_kwargs={"k":1})

template = """> <instructions>: {text} </instructions>

--------------------
--------------------
--------------------

The above instructions represent the best practices for prompt engineering in Large Language Models (LLMs), with examples. They may contain various other elements that might be irrelevant to my given objective, but there are parts that align with my goals. Your task is to extract the essence of crafting a good prompt specifically tailored to my objective. Please assist me in creating an effective prompt template for my purpose."

My objective is enclosed within triple backticks ```.
return your response in the following format:
prompt: ...

```{objective}```

Please format my obective in good prompt by following above instructions that are relevant to my objective.Please do your best it is very important to my career.
"""
prompt = PromptTemplate.from_template(template)

retrieval_chain = (
        {
        "text": itemgetter("objective") | retriever,
        "objective": itemgetter("objective")
    }
    | prompt
    | google
    | StrOutputParser()
)
