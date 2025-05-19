from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import threading
from datetime import datetime, timezone, timedelta
import sys
from flask import Flask, render_template, request, jsonify, session
from flask import redirect, url_for
import uuid

from sklearn.feature_extraction.text import TfidfVectorizer


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llama import get_llama_response
from models.finetuned import get_finetuned_response
from models.rag import get_rag_response

from collections import defaultdict


app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, supports_credentials=True) # Allow CORS for all routes

# File paths
USER_DATA_FILE = 'users.json'
UPLOAD_FOLDER = 'uploads'
QUERY_FILE = 'queries.json'
FEEDBACK_FILE = 'feedback.json'
CONVERSATION_FILE = 'conversations.json'  # New file for storing conversations
app.secret_key = 'your_secret_key_here'  

file_lock = threading.Lock()

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Ensure queries.json exists
if not os.path.exists(QUERY_FILE):
    with open(QUERY_FILE, 'w') as f:
        json.dump([], f)

# Ensure conversations.json exists
if not os.path.exists(CONVERSATION_FILE):
    with open(CONVERSATION_FILE, 'w') as f:
        json.dump({}, f)

# ---------- Utility Functions ----------

def load_json_file(filepath):
    with file_lock:
        if not os.path.exists(filepath):
            return [] if filepath != CONVERSATION_FILE else {}  # Return empty list or dict based on file
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if filepath == CONVERSATION_FILE:
                    return data if isinstance(data, dict) else {}  # Ensure it's a dict
                else:
                    return data if isinstance(data, list) else []  # Ensure it's a list
        except json.JSONDecodeError:
            return [] if filepath != CONVERSATION_FILE else {}  # Return empty list or dict based on file


def save_json_file(filename, new_entry):
    # Handle conversations file differently
    if filename == CONVERSATION_FILE:
        save_conversation(new_entry)
        return
        
    # For other files (lists)
    data = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                # Load data from the file
                data = json.load(f)
            except json.JSONDecodeError:
                # If there's a JSON error, treat it as an empty list
                pass

    # Ensure data is a list before appending
    if isinstance(data, list):
        data.append(new_entry)  # Add new entry as a dictionary
    else:
        # If data is not a list, reset it to an empty list and append the entry
        data = [new_entry]

    # Save the updated list back to the file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Data successfully saved to {filename}.")

def save_conversation(conversation_data):
    """Save or update a conversation to the conversations.json file"""
    with file_lock:
        conversations = {}
        if os.path.exists(CONVERSATION_FILE):
            with open(CONVERSATION_FILE, 'r') as f:
                try:
                    conversations = json.load(f)
                except json.JSONDecodeError:
                    conversations = {}
        
        # Get or create conversation ID
        conversation_id = conversation_data.get('conversation_id')
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            conversation_data['conversation_id'] = conversation_id
        
        # Update conversation
        if conversation_id not in conversations:
            conversations[conversation_id] = {
                'user_email': conversation_data.get('email', 'anonymous'),
                'user_type': conversation_data.get('user_type', 'user'),
                'model_type': conversation_data.get('model_type', 'fine-tuned'),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'messages': []
            }
        
        # Add new message to conversation
        if 'query' in conversation_data and 'response' in conversation_data:
            conversations[conversation_id]['messages'].append({
                'role': 'user',
                'content': conversation_data['query'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            conversations[conversation_id]['messages'].append({
                'role': 'assistant',
                'content': conversation_data['response'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
        # Update timestamp
        conversations[conversation_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
        
        # Save back to file
        with open(CONVERSATION_FILE, 'w') as f:
            json.dump(conversations, f, indent=2)
        
        return conversation_id

def extract_top_keywords(feedback_data, top_n=10):
    queries = [item['query'] for item in feedback_data if 'query' in item]
    if not queries:
        return []
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(queries)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    keywords = dict(zip(feature_names, scores))
    top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [kw[0] for kw in top_keywords]

# ---------- Routes ----------
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400

        # Save the file
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        return jsonify({'message': 'File uploaded successfully', 'filename': file.filename})

    except Exception as e:
        print(f"Error in upload_file: {e}")
        return jsonify({'message': f'Upload failed: {str(e)}'}), 500


@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()  # Ensure we are getting JSON data
    
    # Check if the required fields are present
    if 'email' not in data or 'password' not in data:
        return jsonify({'message': 'Missing email or password'}), 400

    email = data['email']
    password = data['password']

    # Load users data
    users = load_json_file(USER_DATA_FILE)

    # Debugging: Check the structure of loaded users data
    print(f"Loaded users: {users}")  # Print loaded users to verify

    # Check if the email is already registered
    if any(user['email'] == email for user in users):
        return jsonify({'message': 'Email already registered'}), 400

    # Add new user
    new_user = {
        'email': email,
        'password': password,
        'role': 'user'
    }

    # Save updated users data
    save_json_file(USER_DATA_FILE, new_user)

    return jsonify({'message': 'Signup successful'}), 200


@app.route('/login', methods=['POST','GET'])
def login():
    if request.method == 'GET':
        return render_template('index.html')
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Missing data'}), 400

    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    users = load_json_file(USER_DATA_FILE)
    user = next((u for u in users if u['email'].lower() == email and u['password'] == password), None)

    if user:
        session['email'] = user['email']
        session['role'] = user['role'] 
        return jsonify({
            'message': 'Login successful',
            'redirect_url': '/admin' if user['role'] == 'admin' else '/user'
        })
    else:
        return jsonify({'message': 'Invalid credentials'}), 401


@app.route('/get_user_data', methods=['GET'])
def get_user_data():
    email = session.get('email', None)
    if email:
        return jsonify({'email': email})
    else:
        return jsonify({'message': 'No user logged in'}), 401


@app.route('/user')
def user_dashboard():
    if 'email' not in session or session.get('role') != 'user':
        return redirect(url_for('login'))
    return render_template('user.html', email=session['email'])

@app.route('/admin')
def admin_dashboard():
    if 'email' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    return render_template('admin.html', email=session['email'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET'])
def signup_page():
    return render_template('signup.html')

# New route to create a new conversation
@app.route('/conversation/new', methods=['POST'])
def new_conversation():
    data = request.get_json()
    user_type = data.get('user_type', 'user')
    user_email = data.get('email', 'anonymous')
    model_type = data.get('model', 'fine-tuned')
    
    # Create a new conversation
    conversation_id = str(uuid.uuid4())
    
    conversation_data = {
        'conversation_id': conversation_id,
        'email': user_email,
        'user_type': user_type,
        'model_type': model_type
    }
    
    save_conversation(conversation_data)
    
    return jsonify({
        'conversation_id': conversation_id
    })

# New route to get conversation history
@app.route('/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    conversations = load_json_file(CONVERSATION_FILE)
    
    if conversation_id not in conversations:
        return jsonify({'message': 'Conversation not found'}), 404
    
    return jsonify(conversations[conversation_id])

# New route to get all user conversations
@app.route('/conversations', methods=['GET'])
def get_user_conversations():
    if 'email' not in session:
        return jsonify({'message': 'Not logged in'}), 401
    
    user_email = session['email']
    conversations = load_json_file(CONVERSATION_FILE)
    
    # Filter conversations for this user
    user_conversations = {
        conv_id: conv_data 
        for conv_id, conv_data in conversations.items() 
        if conv_data.get('user_email') == user_email
    }
    
    return jsonify(user_conversations)

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    query_text = data.get('query')
    user_type = data.get('user_type', 'user')
    user_email = data.get('email', 'anonymous')
    model_type = data.get('model', 'fine-tuned')
    conversation_id = data.get('conversation_id')

    if not query_text:
        return jsonify({'error': 'Query cannot be empty'}), 400

    timestamp = datetime.now(timezone.utc).isoformat()

    # Construct entry
    entry = {
        "query": query_text,
        "timestamp": timestamp,
        "user_type": user_type,
        "email": user_email,
        "model": model_type
    }

    save_json_file(QUERY_FILE, entry)



    try:
        if model_type == "fine_tuned":
            response_text = get_finetuned_response(query_text)
        elif model_type == "rag":
            response = get_rag_response(query_text)
            # print(response)
            if isinstance(response, dict) and 'error' in response:
                response_text=response['error']
                context="No Pdf No Context"
            else :

                response_text=response["answer"]

                if response_text== "I don't know based on the provided context.":
                    context = "⚠️ Not enough context was found for this question."
                else:
                    context=response["context"]

        elif model_type =="llama":
            response_text = get_llama_response(query_text)
        else:
            response_text="Wrong Model selected "
        
        
        # Save to conversation history if conversation_id is provided
        if conversation_id:
            conversation_data = {
                "conversation_id": conversation_id,
                "query": query_text,
                "response": response_text,
                "email": user_email,
                "user_type": user_type,
                "model_type": model_type
            }
            save_conversation(conversation_data)
        
        if model_type == "rag":
            return jsonify({
            "response": response_text,
            "context": context,
            "conversation_id": conversation_id})

        return jsonify({
            "response": response_text,
            "conversation_id": conversation_id
        })
    
    except Exception as e:
        print(f"Error from model: {e}")
        return jsonify({'error': f'Failed to get response from model. {str(e)}'}), 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    required_fields = ['query', 'response', 'feedback', 'email', 'timestamp', 'userType','model_type']

    if not all(field in data for field in required_fields):
        return jsonify({'message': 'Missing feedback fields'}), 400

    feedback_entry = {
        "query": data['query'],
        "response": data['response'],
        "feedback": data['feedback'],
        "email": data['email'],
        "timestamp": data['timestamp'],
        "user_type": data['userType'],
        "model_type": data['model_type'],
        "conversation_id": data.get('conversation_id')  # Add conversation_id if available
    }

    save_json_file(FEEDBACK_FILE, feedback_entry)

    return jsonify({'message': 'Feedback submitted successfully'})

@app.route('/feedback_analysis')
def feedback_analysis():
    return render_template('feedback_analysis.html')

@app.route('/feedback.json')
def get_feedback():
    filepath = os.path.join(app.root_path, 'feedback.json')
    data = load_json_file(filepath)
    return jsonify(data)

@app.route('/get_feedback_data')
def get_feedback_data():
    try:
        with open('feedback.json') as f:
            feedback_data = json.load(f)

        model_filter = request.args.get('model', 'all').lower()
        user_filter = request.args.get('user', 'all').lower()

        def filter_entry(entry):
            model = entry.get('model_type', '').lower()
            user = entry.get('user_type', '').lower()
            return (model_filter == 'all' or model == model_filter) and \
                   (user_filter == 'all' or user == user_filter)

        filtered = [entry for entry in feedback_data if filter_entry(entry)]

        # Suggested Topics (mocked for now)
        suggested_topics = [
        "RAG Model Architecture",
        "Fine-tuning Techniques",
        "LLaMA Evaluation",
        "Reducing Response Time",
        "User Query Optimization"]

        # 1. Model Performance (dummy: avg feedback length as proxy for score)
        avg_model_scores = {
        'rag': 4.9,
        'fine-tuned': 3.9,
        'llama': 3.6}

        # 2. Satisfaction Trend (avg feedback length per day)
        satisfaction_trend = {
            'dates': ["Week 1", "Week 2", "Week 3", "Week 4"],
            'scores': [1.5, 2.0, 3.2, 4.4]  # average satisfaction values
        }

        # 3. Response Time (dummy values per day for now)
        response_time = {
            'rag': 120,
            'fine-tuned': 95,
            'llama': 110
        }

        # 4. User Type Distribution
        user_types = defaultdict(int)
        for entry in feedback_data:
            user = entry.get('user_type', 'unknown').lower()
            user_types[user] += 1

        # 5. Query Categories (simple keyword buckets)
        categories = defaultdict(int)
        for entry in feedback_data:
            q = entry.get('query', '').lower()

            if any(k in q for k in ['bio', 'photosynthesis', 'plant', 'organism']):
                categories['Biology'] += 1
            elif any(k in q for k in ['ai', 'model', 'transformer', 'llama']):
                categories['AI/ML'] += 1
            elif any(k in q for k in ['tech', 'software', 'api']):
                categories['Technology'] += 1
            elif any(k in q for k in ['earth', 'solar', 'water', 'climate']):
                categories['Environment'] += 1
            elif any(k in q for k in ['finance', 'stock', 'market']):
                categories['Finance'] += 1
            else:
                categories['General'] += 1

        return jsonify({
            'feedback': filtered,
            'suggested_topics': suggested_topics,
            'model_performance': avg_model_scores,
            'satisfaction_trend': satisfaction_trend,
            'response_time': response_time,
            'user_type_distribution': user_types,
            'query_categories': categories
        })

    except Exception as e:
        import traceback
        print("Error in /get_feedback_data:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ---------- Run Server ----------

if __name__ == '__main__':
    app.run(debug=True)