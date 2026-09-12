from sentence_transformers import SentenceTransformer, util
import dotenv
dotenv.load_dotenv

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def find_closest(query, sentences):
    embedding = model.encode(sentences)
    query_embedding = model.encode(query)
    # print(embedding,query_embedding)

    closest_index = 0
    closest_score = -1
    for index, vector in enumerate(embedding):
        similarity = util.cos_sim(vector, query_embedding).item()
        if similarity > closest_score:
            closest_score = similarity
            closest_index = index
    return sentences[closest_index]


def main():

    sentences = [
        "That is a happy person",
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]
    
    
    query = 'I love cats'
    embedding = model.encode(sentences)
    query_embedding = model.encode(query)
    result = util.semantic_search(query_embedding,embedding,top_k=2)
    print(result)


if __name__ == '__main__':
    main()
