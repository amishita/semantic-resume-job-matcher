from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import os

resumes = os.listdir("data/resumes")
job_descriptions = os.listdir("data/job_descriptions")

from utils import read_pdf_file, extract_sections

# Function to compute cosine similarity between two vectors
def compute_similarity(text1, text2, model):
    vec1 = model.encode(text1)
    vec2 = model.encode(text2)
    return cosine_similarity([vec1], [vec2])[0][0] # Return the similarity score

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained model for encoding
    
    resume_text = read_pdf_file("data/resumes/C1061.pdf")
    print(extract_sections(resume_text))

    jd_text = read_pdf_file("data/job_descriptions/Volkswagen_JD.pdf")

    for resume in resumes:
        resume_text = read_pdf_file(f"data/resumes/{resume}")
    
        '''
        for jd in job_descriptions:
            jd_text = read_pdf_file(f"data/job_descriptions/{jd}")

            score = compute_similarity(resume_text, jd_text, model)    
            print(f"Cosine Similarity Score of {resume} and {jd}: {score:.2f}")
        '''

if __name__ == "__main__":
    main()
