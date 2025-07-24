# D&D Game Master Agent

An AI-powered Dungeon Master assistant built with LangChain that helps manage tabletop RPG sessions by providing task difficulty assessment and location information retrieval.

## Features

- **Interactive Game Master**: Acts as a game master that responds to player queries
- **Task Assessment**: Automatically categorizes player actions and assigns difficulty levels (1-20) based on D&D attributes:
  - Strength, Dexterity, Intelligence, Wisdom, Charisma
- **Location Information**: Retrieves contextual information about game locations using RAG (Retrieval-Augmented Generation)
- **PDF Integration**: Ingests PDF documents (like adventure modules) into a vector database for location queries

## Architecture

The project uses a multi-agent architecture with specialized agents:

- **Main Agent**: Orchestrates the game master experience using ReAct prompting
- **Task Agent**: Analyzes player actions and assigns categories/difficulty
- **Location Agent**: Retrieves information about game locations from ingested documents

## Setup

### Prerequisites

- Python 3.11
- OpenAI API key
- Pinecone API key and index

### Installation

1. Install dependencies using pipenv:
```bash
pipenv install
pipenv shell
```

Or using pip:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=your_pinecone_index_name
```

### Document Ingestion

Before running the main application, ingest your adventure documents:

```bash
python ingestion.py
```

This will:
- Load the PDF document (`dyson-logos-challenge-of-the-frog-idol.pdf`)
- Split it into chunks
- Create embeddings using OpenAI
- Store in Pinecone vector database

## Usage

Run the interactive game master:

```bash
python main.py
```

The system will prompt for input and respond as a game master. You can:

- Ask about performing specific tasks (e.g., "I want to climb the wall")
- Inquire about locations (e.g., "What's in the temple?")
- Get contextual information from the ingested adventure module

## Example Interactions

**Task Assessment:**
```
Player: "I want to pick the lock on the chest"
GM: This is a Dexterity-based task with difficulty 12.
```

**Location Information:**
```
Player: "Tell me about the frog idol chamber"
GM: [Retrieves relevant information from the ingested PDF]
```

## Dependencies

- **LangChain**: Core framework for building the agent system
- **OpenAI**: Language model and embeddings
- **Pinecone**: Vector database for document storage and retrieval
- **PyPDF**: PDF document processing
- **python-dotenv**: Environment variable management

## File Structure

```
agent-example/
├── main.py              # Main application entry point
├── ingestion.py         # Document ingestion script
├── agents/
│   ├── taskagent.py     # Task categorization and difficulty assessment
│   └── locationagent.py # Location information retrieval
└── dyson-logos-challenge-of-the-frog-idol.pdf  # Sample adventure module
```