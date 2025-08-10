# AI-PDF-Assistant 🤖📄

An intelligent PDF chatbot powered by Google Gemini AI and LangChain that allows you to upload PDF documents and ask questions about their content. The chatbot uses advanced natural language processing to provide accurate, context-aware responses based on your uploaded documents.

## ✨ Features

- **📄 PDF Processing**: Upload and process multiple PDF files simultaneously
- **🤖 AI-Powered Q&A**: Get intelligent answers using Google Gemini AI
- **🔍 Semantic Search**: Advanced vector search using FAISS for relevant context
- **💬 Interactive Chat**: Clean Streamlit interface with chat history
- **📚 Document Memory**: Maintains context across multiple questions
- **🎯 Accurate Responses**: Only answers based on uploaded document content
- **📱 User-Friendly**: Intuitive web interface with real-time processing

## 🛠️ Technology Stack

- **AI Model**: Google Gemini 2.5 Flash
- **Framework**: LangChain
- **Vector Database**: FAISS
- **Web Interface**: Streamlit
- **PDF Processing**: PyPDF2
- **Embeddings**: Google Generative AI Embeddings

## 📋 Prerequisites

Before running this project, make sure you have:

- Python 3.8 or higher
- Google Gemini API key
- Internet connection for API calls

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/AI-PDF-Assistant.git
cd AI-PDF-Assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root and add your Google Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

**How to get your Google Gemini API key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key to your `.env` file

## 🎯 Usage

### Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

### How to Use the Chatbot

1. **Upload PDF Files:**
   - Click on the sidebar menu
   - Use the file uploader to select one or more PDF files
   - Click "Submit & Process" to analyze the documents

2. **Ask Questions:**
   - Type your question in the text input field
   - Press Enter or click to submit
   - Get instant, context-aware answers

3. **View Chat History:**
   - All conversations are saved in the session
   - Processed files are listed for reference

### Example Questions You Can Ask

- "What is the main topic of this document?"
- "Summarize the key points from the PDF"
- "What are the educational qualifications mentioned?"
- "List all the skills mentioned in the resume"
- "What experience does the person have?"
- "What are the project details?"

## 📁 Project Structure

```
AI-PDF-Assistant/
├── app.py                # Main Streamlit application
├── gemini_chat.py        # Basic Gemini chat (fallback)
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .env                 # Environment variables (create this)
├── .gitignore          # Git ignore rules
└── faiss_index/        # Vector database (auto-generated)
```

## 🔧 Configuration

### Customizing the Chatbot

You can modify the following parameters in `app.py`:

- **Chunk Size**: Change `chunk_size=10000` in `get_text_chunks()` function
- **Chunk Overlap**: Modify `chunk_overlap=1000` for better context
- **Temperature**: Adjust `temperature=0.3` in the model configuration
- **Model**: Switch between different Gemini models

### Advanced Settings

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,     
    chunk_overlap=1000    
)

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3        
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Error with FAISS:**
   ```bash
   pip install langchain-community
   ```

2. **API Key Error:**
   - Ensure your `.env` file exists and contains the correct API key
   - Check that the API key is valid and has sufficient quota

3. **PDF Processing Issues:**
   - Ensure PDFs are not password-protected
   - Check that PDFs contain extractable text (not just images)

4. **Memory Issues:**
   - Reduce chunk size for large documents
   - Process fewer PDFs at once


## 🔒 Privacy & Security

- **Local Processing**: PDFs are processed locally on your machine
- **No Data Storage**: Uploaded files are not permanently stored
- **API Security**: Only document content is sent to Google's API for processing
- **Session Data**: Chat history is cleared when you close the browser


## 📄 License

This project is licensed under the MIT License
