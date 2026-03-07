from PyPDF2 import PdfReader
import re

def read_pdf_file(path: str):
    reader = PdfReader(path)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text() + "\n"
        
    return text

def extract_sections(text: str) -> dict:
    sections = {}

    # Define regex patterns for common resume sections
    patterns = {
        "education": r"(education|academic background|qualifications)",
        "work_experience": r"(work experience|employment history)",
        "skills": r"(skills|core competencies|technical skills)",
        "tech_stack": r"(tech stack|technologies|tools)"
    }

    text_lower = text.lower()
    found = {} # To track which sections have been found

    # Search for each section in the text using the defined patterns
    for section, pattern in patterns.items():
        match = re.search(pattern, text_lower)
        if match:
            found[section] = match.start()

    # Sort found sections by their position in the text
    sorted_sections = sorted(found.items(), key=lambda x: x[1])

    # Extract section content based on the positions of the found sections
    sections = {}
    for i, (section, start) in enumerate(sorted_sections):
        end = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
        sections[section] = text[start:end].strip()

    return sections 
    