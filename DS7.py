# 7) Text Analytics
# 1. Extract Sample document and apply following document preprocessing methods: Tokenization, POS Tagging, stop words removal, Stemming and Lemmatization.
# 2. Create representation of documents by calculating Term Frequency and Inverse DocumentFrequency.
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np

# Ensure NLTK resources are downloaded
def ensure_nltk_resources():
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng'),
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet')
    ]
    for resource_name, resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource: {resource_name}")
            try:
                nltk.download(resource_name)
            except Exception as e:
                print(f"Failed to download {resource_name}: {e}")
                print(f"Please manually download {resource_name}.zip from https://github.com/nltk/nltk_data/tree/gh-pages/packages/ and extract to C:\\Users\\Anil Abhange\\AppData\\Roaming\\nltk_data")
                print("For example, extract averaged_perceptron_tagger.zip to C:\\Users\\Anil Abhange\\AppData\\Roaming\\nltk_data\\taggers\\")
                exit(1)

# Check resources & show them
try:
    ensure_nltk_resources()
except Exception as e:
    print(f"Error setting up NLTK resources: {e}")
    exit(1)
# Sample documents
documents = [
    "The Iris dataset contains measurements of sepal length and width for three species.",
    "Iris setosa, versicolor, and virginica are classified based on petal size.",
    "Machine learning models use Iris data for classification tasks.",
    "Sepal and petal dimensions help distinguish Iris species effectively."
]
# Step 1: Document Preprocessing
def preprocess_document(doc):
    try:
        # Tokenization
        tokens = word_tokenize(doc.lower())
        print(f"\nTokens for document: {doc}")
        print(tokens)
        
        # POS Tagging (Part-of-Speech tagging)
        pos_tags = pos_tag(tokens, lang='eng')
        print("\nPOS Tags:")
        print(pos_tags)
        
        # Stop Words Removal e.g ['the', 'and', 'are']
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
        print("\nAfter Stop Words Removal:")
        print(filtered_tokens)
        
        # Stemming e.g "running" → "run", "better" → "better"
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        print("\nStemmed Tokens:")
        print(stemmed_tokens)
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        print("\nLemmatized Tokens:")
        print(lemmatized_tokens)
        
        return ' '.join(lemmatized_tokens)
    except LookupError as e:
        print(f"Error preprocessing document '{doc}': {e}")
        print("Please ensure NLTK resources are downloaded (see instructions above).")
        return ''
    except Exception as e:
        print(f"Unexpected error preprocessing document '{doc}': {e}")
        return ''

# Preprocess all documents
print("=== Document Preprocessing ===")
preprocessed_docs = [preprocess_document(doc) for doc in documents]
preprocessed_docs = [doc for doc in preprocessed_docs if doc]
# Step 2: TF-IDF Representation
if preprocessed_docs:
    print("\n=== TF-IDF Representation ===")
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
        terms = vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=terms, index=[f'Document {i+1}' for i in range(len(preprocessed_docs))])
        print("\nTF-IDF Matrix:")
        print(tfidf_df.round(4))

        # Manual TF for Document 1
        def calculate_tf(doc):
            tokens = word_tokenize(doc.lower())
            tokens = [token for token in tokens if token.isalpha()]
            token_counts = Counter(tokens)
            total_tokens = len(tokens)
            return {term: count/total_tokens for term, count in token_counts.items()}

        print("\nTerm Frequency (TF) for Document 1:")
        tf_doc1 = calculate_tf(documents[0])
        for term, tf_value in sorted(tf_doc1.items()):
            print(f"{term}: {tf_value:.4f}")

        # Manual IDF
        def calculate_idf(docs):
            term_doc_count = {}
            total_docs = len(docs)
            for doc in docs:
                tokens = set(word_tokenize(doc.lower()))
                for token in tokens:
                    if token.isalpha():
                        term_doc_count[token] = term_doc_count.get(token, 0) + 1
            return {term: np.log(total_docs / (count + 1)) + 1 for term, count in term_doc_count.items()}

        print("\nInverse Document Frequency (IDF):")
        idf_values = calculate_idf(documents)
        for term, idf_value in sorted(idf_values.items()):
            print(f"{term}: {idf_value:.4f}")
    except Exception as e:
        print(f"Error computing TF-IDF: {e}")
else:
    print("No documents preprocessed successfully. Please ensure NLTK resources are available.")