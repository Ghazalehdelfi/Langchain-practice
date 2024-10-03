# Langchain-practice
This repo is a practice code for learning to work with langchain and RAG implementation
the idea is to have a primitive dungeon master implemented by AI with the ReAct framework
the adventure guide will be broken into chunks, embedded and uploaded to our pinecone vectore db via `ingestion.py`
the main engine will have 2 agents, one agent to retrieve location information through RAG and another one to determine difficulty and category of the action the player wants to perform
