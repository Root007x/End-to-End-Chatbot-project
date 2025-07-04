# End-to-End-Chatbot-project

# Bangladesh Constitutional Assistant

A web-based AI chatbot for exploring and understanding the Constitution of Bangladesh.

## Features

- Natural language Q&A about the Constitution of Bangladesh
- Session-based chat memory

## How to Use

### 1. Requirements

- Python 3.8+
- pip
- (Recommended) Virtual environment

### 2. Installation

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables for Pinecone and HuggingFace API keys. You can use a `.env` file:
   ```env
   PINECONE_API_KEY=your_pinecone_key
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
   ```

### 3. Running the App

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser and go to: [http://localhost:5000](http://localhost:5000)

### 4. Using the Chatbot

- Type your question about the Constitution of Bangladesh in the chat box and press Send.
- The assistant will reply with relevant information.
- Your chat history is remembered during your session.
- When you close or reload the page, your session memory is cleared for privacy.

## License

See [LICENSE](LICENSE).

---

For any issues or suggestions, please open an issue or contact the maintainer.
