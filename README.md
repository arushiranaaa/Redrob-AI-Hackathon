#Redrob AI Campus Hackathon: Resume Matching Engine
Project Description
*This repository contains a Python-based matching engine developed for the Redrob AI Campus Hackathon. The program ranks candidates from Indian universities against Job Descriptions (JDs) from Korean tech companies using specific NLP formulas.

#Technical Implementation
*This solution was built using only Python Standard Libraries, strictly following the competition constraints that prohibit external libraries like pandas or numpy.  
*Workflow Stages:Skill Normalization: Raw skill strings are lowercased and mapped using the official SKILL_ALIASES dictionary.  
*Token Filtering: Tokens not present in the alias map are discarded, and the skill "R" is mapped to an empty string.  
*Deduplication: Each canonical skill appears only once per resume profile.  
*Vocabulary Construction: A shared vocabulary is built from resume skills and sorted alphabetically.  
*TF-IDF Calculation:TF (Term Frequency): $TF = 1/N$ (where $N$ is the number of unique skills).  
*IDF (Inverse Document Frequency): $IDF = \ln(10/df)$ using natural logarithms with no smoothing. 

#Similarity & Ranking:Cosine Similarity: 
*Calculated between Resume TF-IDF vectors and JD Binary vectors.  
*Tie-Breaking: If scores match, candidates are sorted alphabetically by name.  
*Rounding: Final scores are rounded to exactly 2 decimal places. 

#How to Run:
*Ensure you have Python 3 installed.

*Run the script via terminal: python resume_matching.py.

*The output will display the Top 3 matching candidates per Job Description.
