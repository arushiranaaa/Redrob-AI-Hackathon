import math
import re

# Implementation Workflow: Data Setup
# Defining Resume and Job Description Datasets
resume_data = [
    {"ID": "01", "Name": "Arjun Sharma", "Raw Skills": "Pyhton, Machine Learning, SQL, pandas, numpy, Deep-learning"},
    {"ID": "02", "Name": "Priya Nair", "Raw Skills": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"},
    {"ID": "03", "Name": "Rahul Gupta", "Raw Skills": "Java, Spring Boot, MySql, Microservices, Docker, kubernates"},
    {"ID": "04", "Name": "Sneha Patel", "Raw Skills": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"},
    {"ID": "05", "Name": "Vikram Singh", "Raw Skills": "C++, Algoritms, Data Structure, competitive programming, python"},
    {"ID": "06", "Name": "Ananya Krishnan", "Raw Skills": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"},
    {"ID": "07", "Name": "Karan Mehta", "Raw Skills": "Python, Sklearn, XGboost, feature engineering, SQL, tableau"},
    {"ID": "08", "Name": "Deepika Rao", "Raw Skills": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"},
    {"ID": "09", "Name": "Aditya Kumar", "Raw Skills": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"},
    {"ID": "10", "Name": "Meera lyer", "Raw Skills": "python, R, statistics, ML, regression, clustering, Power-Bl"}
]

jd_data = [
    {"ID": "JD-1", "Company": "Kakao (ML Engineer)", "Required Skills": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization", "Preferred Skills": "NLP, BERT, Feature Engineering, Statistics"},
    {"ID": "JD-2", "Company": "Naver (Backend Engineer)", "Required Skills": "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes", "Preferred Skills": "REST API, CI/CD, Redis"},
    {"ID": "JD-3", "Company": "Line (Frontend Engineer)", "Required Skills": "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS", "Preferred Skills": "Node.js, GraphQL, Redux, Jest, AWS"}
]

# Official SKILL_ALIASES Mapping (Do not modify)
skill_aliases = {
    "python": "python", "pyhton": "python", "java": "java", "javascript": "javascript",
    "javascrpit": "javascript", "js": "javascript", "typescript": "typescript",
    "typescrpit": "typescript", "c++": "cpp", "cpp": "cpp", "r": "", "kotlin": "kotlin",
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning", "deeplearning": "deep_learning",
    "deep learning": "deep_learning", "deep-learning": "deep_learning", "tensorflow": "tensorflow",
    "pytorch": "pytorch", "keras": "keras", "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering", "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering", "data-viz": "data_visualization",
    "data visualization": "data_visualization", "data viz": "data_visualization",
    "matplotlib": "data_visualization", "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization", "pandas": "pandas",
    "numpy": "numpy", "react": "react", "reacts": "react", "reactjs": "react", "vue": "vue",
    "vue.js": "vue", "vuejs": "vue", "redux": "redux", "tailwind": "tailwind", "html/css": "html_css",
    "html css": "html_css", "html": "html_css", "css": "html_css", "jest": "jest",
    "graphql": "graphql", "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask", "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api", "microservices": "microservices",
    "sql": "sql", "mysql": "mysql", "mysq": "mysql", "postgresql": "postgresql",
    "postgres": "postgresql", "mongodb": "mongodb", "redis": "redis", "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd", "aws": "aws", "android": "android",
    "firebase": "firebase", "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming", "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma"
}

# Step 1: Skill Normalization Logic
def normalize_and_deduplicate(raw_string):
    # Lowercase tokens and split raw strings using commas
    tokens = [t.strip().lower() for t in raw_string.split(',')]
    normalized = []
    for token in tokens:
        # Apply alias mapping and discard unknown tokens
        if token in skill_aliases:
            canonical = skill_aliases[token]
            if canonical != "": # Discard "r" tokens
                normalized.append(canonical)
    
    # Step 2: Deduplication - unique canonical skills per resume
    return sorted(list(set(normalized)))

# Step 3: Vocabulary Construction (Alphabetical Sort)
processed_resumes = [normalize_and_deduplicate(r["Raw Skills"]) for r in resume_data]
vocabulary = sorted(list(set(skill for resume in processed_resumes for skill in resume)))

# Step 4: TF-IDF Vector Construction
def get_tfidf_vectors(resumes, vocab):
    n_resumes = len(resumes)
    vectors = []
    # Calculate DF (Document Frequency) per skill 
    df_counts = {skill: sum(1 for res in resumes if skill in res) for skill in vocab}
    
    for resume in resumes:
        vector = []
        n_unique_skills = len(resume) # Total unique skills
        for skill in vocab:
            if skill in resume:
                # TF Calculation (1 / N after deduplication)
                tf = 1 / n_unique_skills
                # IDF Calculation (Natural Log, no smoothing)
                idf = math.log(n_resumes / df_counts[skill])
                vector.append(tf * idf)
            else:
                vector.append(0.0)
        vectors.append(vector)
    return vectors

resume_tfidf_vectors = get_tfidf_vectors(processed_resumes, vocabulary)

# Building Binary JD Vectors over the same vocabulary 
def get_jd_vectors(jds, vocab):
    vectors = []
    for jd in jds:
        combined_raw = jd["Required Skills"] + ", " + jd["Preferred Skills"]
        jd_skills = normalize_and_deduplicate(combined_raw)
        # Create binary vectors for JDs 
        vector = [1 if skill in jd_skills else 0 for skill in vocab]
        vectors.append(vector)
    return vectors

jd_binary_vectors = get_jd_vectors(jd_data, vocabulary)

# Cosine Similarity Calculation
def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a**2 for a in v1))
    mag2 = math.sqrt(sum(b**2 for b in v2))
    if mag1 == 0 or mag2 == 0: return 0.0
    return dot_product / (mag1 * mag2)

# Final Similarity & Ranking per JD
for i, jd in enumerate(jd_data):
    results = []
    for j, resume in enumerate(resume_data):
        score = cosine_similarity(resume_tfidf_vectors[j], jd_binary_vectors[i])
        results.append({"name": resume["Name"], "score": round(score, 2)})
    
    # Ties broken alphabetically by candidate name
    results.sort(key=lambda x: (-x["score"], x["name"]))
    
    # Printing Output in Expected Format 
    print(f"{jd['ID']}")
    print(f"{jd['Company']}")
    top_3 = [f"{r['name']} ({r['score']:.2f})" for r in results[:3]] # 
    print(", ".join(top_3))
    print()