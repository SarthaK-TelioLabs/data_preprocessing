import json
import nltk
import string
import re

# Sample Jira issue ticket from API
ticketJira = '''
{
  "issue_id": "data_pre_project",
  "description": "Users are reporting issues with the application's login process. When attempting to log in with their credentials, the system fails to authenticate and displays an error message. This problem seems to occur intermittently across different devices and browsers. The login button appears unresponsive in some instances, preventing users from accessing their accounts. We need to investigate and resolve this issue promptly to ensure seamless user experience and maintain customer satisfaction.",
  "status": "Open",
  "user_id": "john.doe",
  "created_date": "2023-06-15"
}
'''
#loading the json 
ticket=json.loads(ticketJira)
#downloading all nltk requirements
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')
#Now create a function to extract the string and preprocess the data using NLTK

def data_preprocess(ticket):
    #extracting the description from json ticket
    description = ticket.get('description')
    # Lowercasing
    description = description.lower()
    
    # Remove HTML codes (if any)
    description = re.sub(r"<[^>]+>", "", description)
    
    # Tokenization and remove punctuation
    tokens = nltk.word_tokenize(description)
    filtered_tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    filtered_tokens = [token for token in filtered_tokens if token.lower() not in stopwords]
    
    # Lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # Part of Speech tagging so we can use it later
    tagged_tokens = nltk.pos_tag(lemmatized_tokens)
    
    return tagged_tokens

#Output example
processed_tokens= data_preprocess(ticket)
print("Processed tokens:")
print(processed_tokens)
