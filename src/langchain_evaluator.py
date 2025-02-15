from typing import Any

import nest_asyncio
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langsmith.schemas import Example, Run
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.base import Metric, MetricWithEmbeddings, MetricWithLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.metrics import answer_relevancy, context_precision
from langsmith.evaluation import evaluate

from chain.langchain_rag_simple import chain

nest_asyncio.apply()


class RagasMetricEvaluator:
    def __init__(self, metric: Metric, llm: BaseChatModel, embeddings: Embeddings):
        self.metric = metric

        # LLMとEmbeddingsをMetricに設定
        if isinstance(metric, MetricWithLLM):
            self.metric.llm = LangchainLLMWrapper(llm)
        if isinstance(metric, MetricWithEmbeddings):
            self.metric.embeddings = LangchainEmbeddingsWrapper(embeddings)

    def evaluate(self, run: Run, example: Example) -> dict[str, Any]:
        context_strs = [doc.page_content for doc in run.outputs["contexts"]]

        # Ragasの評価メトリクスのscoreメソッドでスコアを算出
        score = self.metric.score(
            {
                "question": example.inputs["question"],
                "answer": run.outputs["answer"],
                "contexts": context_strs,
                "ground_truth": example.outputs["ground_truth"],
            },
        )
        return {"key": self.metric.name, "score": score}


metrics = [context_precision, answer_relevancy]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, request_timeout=60)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

evaluators = [
    RagasMetricEvaluator(metric, llm, embeddings).evaluate for metric in metrics
]


def predict(inputs: dict[str, Any]) -> dict[str, Any]:
    question = inputs["question"]
    output = chain.invoke(question)
    return {
        "contexts": output["context"],
        "answer": output["answer"],
    }


evaluate(
    predict,
    data="agent-book",
    evaluators=evaluators,
)
