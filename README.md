# BERT Interactive QA Chatbot
This is a conversational chatbot with login and registration functionality that leverages a fine-tuned BERT model for answering user questions. The chatbot is built using Flask and integrates with a web application for user authentication and personalized responses.

## Technology
- Python 3.7+
- Flask
- Hugging Face Transformers
- SQLAlchemy
- WTForms

## Features
- User registration and login
- Personalized responses based on user profile
- Fine-tuned BERT model for answering user questions
- Database storage of user profiles and chatbot responses

## Installation
1. Clone this repository.
2. Create a virtual environment and activate it. `python3 -m venv venv` and `source venv/bin/activate`.
3. Install the required packages using `pip install -r requirements.txt`.
4. Copy the `.env.example` file to `.env` and update the environment variables.
5. Set environment variables for the Flask application. `export FLASK_APP=wsgi.py` windows `set FLASK_APP=wsgi.py`.
6. Migrate the database.Delete the `migrations` folder in your project directory, if it exists
    ```bash
    flask db init
    flask db migrate -m "Initial migration"
    flask db upgrade
    ```
7. Run the Flask application using `flask run`.
8. Access the chatbot at http://localhost:5000/.

## Usage
1. Register for a new account or log in with an existing one.
2. Enter your query in the chatbot input field.
3. Receive a personalized response based on your user profile and the BERT model's understanding of your query.

## Future Improvements
- Integration with external APIs for more diverse and accurate responses
- Deployment on a cloud platform for wider accessibility
- Integration with analytics and logging tools for performance monitoring