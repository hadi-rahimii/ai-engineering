import hashlib
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import dotenv
import os
from openai import OpenAI


dotenv.load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv('GROQ_API_KEY'),
    base_url='https://api.groq.com/openai/v1'
)

model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma(
    collection_name='medical_data',
    embedding_function=model,
    persist_directory='./chroma',
    collection_metadata={'hnsw:space': "cosine"}

)


def generate_ids(docs):
    return [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in docs]


def llm(query, context):
    system_prompt = f'''You are a medical information assistant. Answer the user's question using ONLY
    the provided source documents below.

      Rules you must follow:
      1. Base your answer ONLY on the provided sources — do not use outside knowledge
      2. If the sources don't contain enough information to answer, say exactly: "I don't have enough information to answer
      this question."
      3. Cite sources using [Source N] notation whenever you use information from them
      4. If sources contradict each other, mention the contradiction
      5. Be concise — answer in 3-5 sentences unless the question requires more detail
      6. Never speculate or make assumptions beyond what the sources say

      Sources:
       {context}'''

    messages = [{
        'role': 'system', "content": system_prompt
    }, {'role': 'user', 'content': query}]

    response = client.chat.completions.create(
        model='openai/gpt-oss-20b',
        temperature=0.5,
        messages=messages
    )

    return response.choices[0].message.content


def read_pdf(file_name):
    reader = PdfReader(file_name)
    docs = []

    for index, page in enumerate(reader.pages):
        doc = Document(
            page_content=page.extract_text(),
            metadata={"source": file_name, "page": index}
        )
        docs.append(doc)

    return docs


def main():
    
    docs = read_pdf('symptoms.pdf')

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=110, chunk_overlap=10)
    chunks = splitter.split_documents(docs)
    
    
    # //////////////////////////////////////////////////////////////////////////
    # creating vectors 
    # texts = [chunk.page_content for chunk in chunks]
    # vectors = model.embed_documents(texts)
    # print(len(vectors))
    # print(vectors[0])


    #///////////////////////////////////////////////////////////////////////////
    # ids for chunks
    ids = generate_ids(chunks)
    vectorstore.add_documents(chunks , ids = ids)
    # all_docs = vectorstore.get()
    # print(f'all_docs{len(all_docs['documents'])}')
    # print(f'first{all_docs['documents'][0]}')

    while True:
        user_input = input(
            'Ask question (type "q" for terminate )').strip().lower()

        if user_input == 'q':
            print('have a nice day')
            break

        retrieved_chunks = vectorstore.similarity_search_with_score(
            user_input, k=2)
        print(f"retrieved chuks =>>>>>>>>>>>>>>> {retrieved_chunks}")
        print(f"retrieved chuks.doc =>>>>>>>>>>>>>>> {retrieved_chunks.doc}")
        
        parts = []

        for i, (doc, score) in enumerate(retrieved_chunks):
            parts.append(
                f'Source {i+1} | page {doc.metadata.get('page', '?')} | cosine similarity {score: .3f} \n {doc.page_content}'
            )
            
            
        print(f'parts =>>>>>>>>>>>>>>>>>>>>> {parts}')
        context = '\n\n_____\n\n'.join(parts)

        print(llm(user_input, context))

    #     # for i, chunk in enumerate(retrieved_chunks):
    #     #     print(f'__chunk {i} __ \n')
    #     #     print(f'__score {chunk[1]:.3f} __ \n')
    #     #     print(
    #     #         f'__ metadats : source {chunk[0].metadata['source']} __ page : {chunk[0].metadata['page']} __ \n\n')

    #     #     print(chunk[0].page_content[0:300])


if __name__ == '__main__':
    main()
