from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2',
    encode_kwargs = {'normalize_embeddings': True }
)


def read_pdf(file_name):
    reader = PdfReader(file_name)
    print(reader)
    docs=[]
    
    for index , page in enumerate(reader.pages):
        doc = Document(
            page_content=page.extract_text(),
            metadata={"source": file_name, "page": index}
        )
        docs.append(doc)

    return docs

def main ():
    docs = read_pdf('symptoms.pdf')
    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=110, chunk_overlap=10)
    print(f'splitter{splitter}')
    
    chunks = splitter.split_documents(docs)
    print(len(chunks))
    print(f'first chunks{chunks[0]}')
    texts = [chunk.page_content for chunk in chunks]
    vectors = model.embed_documents()
    print(len(vectors))
    print(vectors[0])
    
    # model.embed_documents()
    # print(len(docs))
    # print(docs[0])



if __name__ == '__main__':
    main()