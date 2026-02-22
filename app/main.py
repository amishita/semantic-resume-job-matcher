from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

def read_text_file(path: str):
    with open(path, 'r') as file:
        return file.read()

# Function to compute cosine similarity between two vectors
def compute_similarity(text1, text2, model):
    vec1 = model.encode(text1)
    vec2 = model.encode(text2)
    return cosine_similarity([vec1], [vec2])[0][0] # Return the similarity score

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained model for encoding
    
    resume_text = "I specialize in frontend development using React and CSS."

    jd_text = read_text_file("data/job_descriptions/ml_engineer.txt")

    score = compute_similarity(resume_text, jd_text, model)    
    print(f"Cosine Similarity Score: {score:.2f}")

if __name__ == "__main__":
    main()
