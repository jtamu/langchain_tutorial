from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from db.chroma_db import db

prompt = ChatPromptTemplate.from_template(
    '''\
    以下の文脈だけを踏まえて質問に回答してください。

    文脈："""
    {context}
    """

    質問：{question}
    '''
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = db.as_retriever()

chain = RunnableParallel(
    {
        "question": RunnablePassthrough(),
        "context": retriever,
    }
).assign(answer=prompt | model | StrOutputParser())
