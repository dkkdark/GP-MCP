
## Steps to set up the project:

#### Follow these steps to install dependencies, set up your environment, and run the project.

1. Clone the project

```
git clone <your-repo-url>
cd <your-project-folder>
```

2. Create and activate a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

3. Install the requirements

```
pip install -r requirements.txt
```

4. Create a .env file in the project root and add your OpenAI API key:

```
touch .env
```

Then open .env and add the following line:

```
OPENAI_API_KEY=your-openai-api-key-here
```

5. Place your learning materials (in docx format!) in the ./data folder. Assigments in the tasks folder and lectures in the materials


6. In Terminal 1, run the server:

```
python -m server.server
```

This loads and the documents from the data folder. First time it takes some time to create vector db. After this step teaching_chroma_db will be created as well as chuncks.txt file. 

In the txt file you can see chuncks system created. They are contextual, one chunck should represent one piece of meaningfull information.

You should see: "Uvicorn running on http://0.0.0.0:8000"

7. In a separate Terminal 2, run the Streamlit app:

If you want to chat by yourself, run:

```
streamlit run chat_client/main_manual_chat.py
```

If you want to run student simulation (where student defined by a prompt get_student_simulation_prompt. You can find it in prompts.py), run:

```
streamlit run chat_client/main_student_simulation_chat.py
```

You should be redirected to a browser

## Project structure description

<b>common/</b>
Contains shared configuration and prompt templates used by both the server and the chat client.

<b>server/</b>
This folder contains all backend logic, including document processing and tool definitions.

1. server.py implements a single MCP tool that retrieves relevant material based on a user query and launches the MCP server (https://github.com/modelcontextprotocol/python-sdk).
2. document_processor.py handles RAG logic, chunk creation and filtering, vector database setup and loading

<b>chat_client/</b>
Implements the front-end chat interface and client-side logic.

1. main_manual_chat.py & main_student_simulation_chat.py - streamlit-based chat interfaces
2. client.py: Handles communication with the MCP server
3. tools.py: Tool configuration and registration for the client
4. agent.py defines LangChain agents and the graph that connects them, enabling structured multi-step reasoning and interaction.
