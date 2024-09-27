from langchain_community.document_loaders import PDFPlumberLoader
loader = PDFPlumberLoader(r"F:\Project_RAG\2022.lateraisse-1.2.pdf")
docs = loader.load()

# Check the number of pages
print("Number of pages in the PDF:",len(docs))

# Load the random page content
# print(docs[1].page_content)

# Output
"""
Leaf Diseases Caused By Fungi and Bacteria
Leaf Spots
Bacteria or fungi can cause leaf spots that vary in size, shape, and color. Usually the spot has a distinct margin and may be surrounded by a yellow
halo. A fungal leaf spot nearly always has a growth of some type in the spot, particularly in damp weather. It may be a tiny pimple-like structure or a
moldy growth of spores. Often the structures are visible through a hand lens. Nearby diseased areas may join to form irregular "blotches."
photo: Paul Bachi, University of Kentucky Research and Education Center, Bugwood.org
Septoria brown spot is a common fungal disease of soybeans. It causes small angular red-brown spots to develop on upper and lower surfaces of
trifoliate leaves 2 to 3 weeks after planting. Numerous spots will cause leaves to yellow and drop. The disease will develop many irregular, tan lesions
on trifoliate leaves that later turn dark brown. Individual spots will frequently coalesce to form large blackish-brown blotches.
Defoliation from the bottom to the top of severely diseased trifoliate leaves is common during wet seasons. Early season brown spot will appear
annually in almost every field in Kentucky. Late-season brown spot is much more variable in occurrence and severity.
The fungus survives from season to season in crop debris and seed. Warm, moist weather promotes the sporulation of the fungus; the spores are
spread by wind and rain. Hot, dry weather can stop disease development.
Leaf Blights
Leaf blights generally affect larger leaf areas and are more irregular than leaf spots.
photo: Margaret McGrath, Cornell University, Bugwood.org
Northern corn leaf blight (NCLB), caused by a fungus, first develops on the lower leaves of corn plants that are from waist to shoulder high. The
telltale sign of northern corn leaf blight is the 1-to-6 inch long cigar-shaped gray-green to tan-colored lesions on the lower leaves. As the disease
develops, the lesions spread to all leafy structures.
Wet weather and moderate temperatures favor NCLB. Symptoms can be confused with bacterial wilt, especially late in the season.
"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings

text_splitter = SemanticChunker(HuggingFaceEmbeddings())
documents = text_splitter.split_documents(docs)

# Check number of chunks created
print("Number of chunks created: ", len(documents))
# Output
"""
Number of chunks created:  23
"""

# Printing first few chunks
# for i in range(len(documents)):
#     print()
#     print(f"CHUNK : {i+1}")
#     print(documents[i].page_content)


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Instantiate the embedding model
embedder = HuggingFaceEmbeddings()

# Create the vector store
vector = FAISS.from_documents(documents, embedder)


# Input
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retrieved_docs = retriever.invoke("tell me what is bias?")
# print(retrieved_docs)


from langchain_community.llms import Ollama

# Define llm
llm = Ollama(model="mistral-nemo")

from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate

prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences.

Context: {context}

Question: {question}

Helpful Answer:"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

llm_chain = LLMChain(
                  llm=llm,
                  prompt=QA_CHAIN_PROMPT,
                  callbacks=None,
                  verbose=True)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
                  llm_chain=llm_chain,
                  document_variable_name="context",
                  document_prompt=document_prompt,
                  callbacks=None,
              )

qa = RetrievalQA(
                  combine_documents_chain=combine_documents_chain,
                  verbose=True,
                  retriever=retriever,
                  return_source_documents=True,
              )
print(qa("What effect gender has?")["result"])