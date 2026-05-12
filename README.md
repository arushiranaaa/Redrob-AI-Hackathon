# Redrob AI Campus Hackathon: Resume Matching Engine

## Project Description
[cite_start]This repository contains a Python-based matching engine developed for the Redrob AI Campus Hackathon. [cite: 1] [cite_start]The program ranks candidates from Indian universities against Job Descriptions (JDs) from Korean tech companies using specific NLP formulas. [cite: 9, 10]

## Technical Implementation
[cite_start]This solution was built using only **Python Standard Libraries**, strictly following the competition constraints that prohibit external libraries like `pandas` or `numpy`. [cite: 220]

### Workflow Stages:
* [cite_start]**Skill Normalization**: Raw skill strings are lowercased and mapped using the official `SKILL_ALIASES` dictionary. [cite: 23, 25]
* [cite_start]**Token Filtering**: Tokens not present in the alias map are discarded, and the skill "R" is mapped to an empty string. [cite: 26, 95]
* [cite_start]**Deduplication**: Each canonical skill appears only once per resume profile. [cite: 31]
* [cite_start]**Vocabulary Construction**: A shared vocabulary is built from resume skills and sorted alphabetically. [cite: 33, 34]

### TF-IDF Calculation:
* [cite_start]**TF (Term Frequency)**: $TF = 1 / N$ (where $N$ is the number of unique skills). 
* [cite_start]**IDF (Inverse Document Frequency)**: $IDF = \ln(10 / df)$ using natural logarithms with no smoothing. [cite: 51, 55, 56]

## Similarity & Ranking
* [cite_start]**Cosine Similarity**: Calculated between Resume TF-IDF vectors and JD Binary vectors. [cite: 60]
* [cite_start]**Tie-Breaking**: If scores match, candidates are sorted alphabetically by name. [cite: 205]
* [cite_start]**Rounding**: Final scores are rounded to exactly 2 decimal places. [cite: 204]

## How to Run
* [cite_start]Ensure you have Python 3 installed. [cite: 11]
* Run the script via terminal: `python resume_matching.py`.
* [cite_start]The output will display the Top 3 matching candidates per Job Description. [cite: 13]
