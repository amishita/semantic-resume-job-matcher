from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to compute cosine similarity between two vectors
def compute_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def main():
    resume_vector = np.array([0.1, 0.2, 0.3, 0.4])
    jd_vector = np.array([0.2, 0.1, 0.4, 0.3])

    score = compute_similarity(resume_vector, jd_vector)    
    print(f"Cosine Similarity Score: {score:.2f}")

if __name__ == "__main__":
    main()
