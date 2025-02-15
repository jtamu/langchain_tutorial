import nest_asyncio
from langchain_community.document_loaders import GitLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loader.load()
print(len(documents))

for document in documents:
    document.metadata["filename"] = document.metadata["source"]

nest_asyncio.apply()

generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model="gpt-4o-mini"),
    critic_llm=ChatOpenAI(model="gpt-4o-mini"),
    embeddings=OpenAIEmbeddings(),
)

testset = generator.generate_with_langchain_docs(
    documents,
    test_size=4,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)

print(testset.to_pandas())

dataset_name = "agent-book"

client = Client()

if client.has_dataset(dataset_name=dataset_name):
    client.delete_dataset(dataset_name=dataset_name)

dataset = client.create_dataset(dataset_name=dataset_name)

inputs = []
outputs = []
metadatas = []

for testset_record in testset.test_data:
    inputs.append(
        {
            "question": testset_record.question,
        }
    )
    outputs.append(
        {
            "contexts": testset_record.contexts,
            "ground_truth": testset_record.ground_truth,
        }
    )
    metadatas.append(
        {
            "source": testset_record.metadata[0]["source"],
            "evolution_type": testset_record.evolution_type,
        }
    )

client.create_examples(
    inputs=inputs, outputs=outputs, metadatas=metadatas, dataset_id=dataset.id
)
