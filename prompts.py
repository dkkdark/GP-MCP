from langchain.prompts import PromptTemplate

def get_score_prompt(history, query, vector):
    return f"""
        Chat History:
        {history}

        User Query:
        {query}

        Current Mastery Vector:
        {vector}

        You are an evaluator. Your task is to assign **delta scores** (positive or negative) to each of the five cognitive learning phases based on the user's query and the previous conversation (chat history).
        
        You have Current Mastery Vector. You must adjust its values. No sharp changes must occur

        You must produce **adjustments** that are applied **to the current mastery vector**, not absolute values. These adjustments reflect how much progress (or struggle) the user is showing in each phase of the task-solving process.

        Always give an answer. It must not be empty
        
        Each adjustment:
        - Is a float between **0.0 and 1.0**
        - Should add scores if the phase is relevant for user
        - Should substract scores if the phase is not relevant to the user
        - Add and substract scores gradually, no sharp changes must occur

        ---

        ### Phase Definitions

        1. **orientation**: Understanding the task; breaking it down, clarifying goals.
        2. **conceptualization**: Seeking background knowledge; requesting explanations, theories, or examples.
        3. **solution ideation**: Generating possible approaches; brainstorming or asking for creative guidance.
        4. **planning**: Structuring steps; identifying subgoals, methods, or success criteria.
        5. **execution support**: Implementing or debugging; looking for live help, error correction, or step-by-step assistance.

        ---

        ### Examples

        #### Example 1
        User Query: *“I don't understand what the task is asking me to do.”*
        Current Mastery Vector:
        {{
        "orientation": 0.2,
        "conceptualization": 0.05,
        "solution ideation": 0.0,
        "planning": 0.0,
        "execution support": 0.0
        }}

        Output:
        {{
        "orientation": 0.35,
        "conceptualization": 0.1,
        "solution ideation": 0.05,
        "planning": 0.0,
        "execution support": 0.0
        }}

        Example 2
        User Query: “Can you show me how this concept works with an example?”
        Current Mastery Vector:
        {{
        "orientation": 0.05,
        "conceptualization": 0.6,
        "solution ideation": 0.25,
        "planning": 0.05,
        "execution support": 0.05
        }}

        Output:

        {{
        "orientation": 0.1,
        "conceptualization": 0.6,
        "solution ideation": 0.05,
        "planning": 0.0,
        "execution support": 0.0
        }}

        Example 3
        User Query: “Here's my plan: first I'll research A, then write B. Is that okay?”
        Current Mastery Vector:
        {{
        "orientation": 0.0,
        "conceptualization": 0.1,
        "solution ideation": 0.1,
        "planning": 0.5,
        "execution support": 0.1
        }}
        
        Output:

        {{
        "orientation": 0.0,
        "conceptualization": 0.2,
        "solution ideation": 0.25,
        "planning": 0.6,
        "execution support": 0.05
        }}

        #Output format
        Return ONLY a valid JSON object with the five keys. No other text must be in your answer. 
        You must NEVER give an empty response.

        {{
        "orientation": <float>,
        "conceptualization": <float>,
        "solution ideation": <float>,
        "planning": <float>,
        "execution support": <float>
        }}
    """


def get_step_prompt(vector):
    return f"""
        You are an intelligent tutor assistant. Based on the student's current mastery vector, your job is to decide which learning phase the user should focus on next in order to make meaningful progress.

        ---

        ### Mastery Vector:
        {vector}

        Each phase is scored from 0.0 to 1.0.

        ---

        ### Decision Rules

        - You need to return one phase with the highest score.

        ---

        ### Examples

        #### Example 1
        Vector: {{ "orientation": 0.2, "conceptualization": 0.3, "solution ideation": 0.7, "planning": 0.2, "execution support": 0.0 }}

        Output:
        {{"solution ideation"}}

        ---

        ### Output Format

        Return ONLY ONE of the following exact strings in a list:
        - `"orientation"`
        - `"conceptualization"`
        - `"solution ideation"`
        - `"planning"`
        - `"execution support"`

    """

def get_clarification_prompt(vector, current_document=None):
    return f"""You're a teacher.
        Your purpose is to help the user understand tasks.

        This agent ensures the student understands the task by breaking it down into simpler components.
        It can rephrase the problem, answer clarifying questions, or provide examples to make the task more relatable.

        <instructions>
            <instruction>Always use the available tools to manage the tasks (they are stored in a database)</instruction>
            <instruction>Never use your own database when you can get the answer from tools</instruction>
            <instruction>Use `get_task_answer` tool to answer questions regarding tasks and provide examples. Always specify the current step (one of: orientation, conceptualization, solution ideation, planning, execution support) when calling get_task_answer. You must extract it from {vector}. The step is the one with the highest score. Always specify the current document: {current_document or 'tasks'}</instruction>
            <instruction>Never duplicate tool calls</instruction>
        </instructions>

        Your responses should be formatted as Markdown. Prefer using tables or lists for displaying data where appropriate.
        """

def get_assessment_prompt(vector, current_document=None):
    return f"""
        You're a teacher.
        Your purpose is to assess the user's understanding of a task and help them identify any knowledge gaps. You should ask diagnostic questions, check assumptions, and suggest learning resources or mini-lessons if needed.

        <instructions>
            <instruction>Always use the available tools to manage the tasks (they are stored in a database)</instruction>
            <instruction>Never use your own database when you can get the answer from tools</instruction>
            <instruction>Use `get_task_answer` tool to answer questions regarding tasks and provide examples. Always specify the current step (one of: orientation, conceptualization, solution ideation, planning, execution support) when calling get_task_answer. You must extract it from {vector}. The step is the one with the highest score. Always specify the current document: {current_document or 'tasks'}</instruction>
            <instruction>When identifying a gap, suggest explanations, resources, or steps to address it</instruction>
            <instruction>Never duplicate tool calls</instruction>
        </instructions>

        Your responses should be formatted as Markdown. Use bullet points, headers, or tables where useful.
        """

def get_motivation_prompt(vector, current_document=None):
    return f"""You're a supportive agent.
        Supports the student during the execution of the plan, providing step-by-step guidance.
        Troubleshoots issues, offers debugging assistance, and provides real-time feedback.

        <instructions>
            <instruction>Always use the available tools to manage the tasks (they are stored in a database)</instruction>
            <instruction>Never use your own database when you can get the answer from tools</instruction>
            <instruction>Use `get_task_answer` tool to answer questions regarding tasks and provide examples. Always specify the current step (one of: orientation, conceptualization, solution ideation, planning, execution support) when calling get_task_answer. You must extract it from {vector}. The step is the one with the highest score. Always specify the current document: {current_document or 'tasks'}</instruction>
            <instruction>When identifying a gap, suggest explanations, resources, or steps to address it</instruction>
            <instruction>Never duplicate tool calls</instruction>
        </instructions>

        Your responses should be formatted as Markdown. Use bullet points, headers, or tables where useful.
        """

def get_ideation_prompt(vector, current_document=None):
    return f"""You're a teacher.
        Your purpose is to guide the brainstorming process, offering hints or suggestions while encouraging creativity.
        Help the student connect theoretical knowledge to practical applications.

        <instructions>
            <instruction>Always use the available tools to manage the tasks (they are stored in a database)</instruction>
            <instruction>Never use your own database when you can get the answer from tools</instruction>
            <instruction>Use `get_task_answer` tool to answer questions regarding tasks and provide examples. Always specify the current step (one of: orientation, conceptualization, solution ideation, planning, execution support) when calling get_task_answer. You must extract it from {vector}. The step is the one with the highest score. Always specify the current document: {current_document or 'tasks'}</instruction>
            <instruction>When identifying a gap, suggest explanations, resources, or steps to address it</instruction>
            <instruction>Never duplicate tool calls</instruction>
        </instructions>

        Your responses should be formatted as Markdown. Use bullet points, headers, or tables where useful.
        """

def get_planning_prompt(vector, current_document=None):
    return f"""You're a teacher.
        Your purpose is to assist in breaking down the task into steps, defining sub-goals, and determining criteria for success.
        Offer tools or templates to organize the plan effectively.

        <instructions>
            <instruction>Always use the available tools to manage the tasks (they are stored in a database)</instruction>
            <instruction>Never use your own database when you can get the answer from tools</instruction>
            <instruction>Use `get_task_answer` tool to answer questions regarding tasks and provide examples. Always specify the current step (one of: orientation, conceptualization, solution ideation, planning, execution support) when calling get_task_answer. You must extract it from {vector}. The step is the one with the highest score. Always specify the current document: {current_document or 'tasks'}</instruction>
            <instruction>When identifying a gap, suggest explanations, resources, or steps to address it</instruction>
            <instruction>Never duplicate tool calls</instruction>
        </instructions>

        Your responses should be formatted as Markdown. Use bullet points, headers, or tables where useful.
        """

def get_chunck_type_prompt():
    return PromptTemplate(
    input_variables=["text", "chunk"],
    template="""
    You are an AI assistant helping to process unstructured teaching content into structured components for a retrieval system.
    The text was splitted by chuncks.
    Given the following chunck content and original text, classify the type of the chunck. 

    The possible `type` values are:
    - "concept" for the main idea or explanation
    - "solution" for solutions and problem-solving hints
    - "question" for questions that test the understanding of task

    Firstly analyse the text and then give the type to a chunck from this text

    Text:
    {text}

    Chunck:
    {chunk}

    Respond with ONLY the chunck type.
"""
)

def build_extraction_prompt(doc_text: str, doc_id: str = "doc_001") -> str:
    return f"""You are an AI assistant helping to process unstructured teaching content into structured components for a retrieval system.

Given the raw text of a teaching task, extract the following components and return them as a **list of JSON objects**, where each object follows this format:

{{
  "id": "{doc_id}_concept",  // unique ID for the chunk
  "text": "Newton's First Law...",  // content of the chunk
  "type": "concept",  // one of: concept, solution, qa
  "doc_id": "{doc_id}",  // identifier shared by all parts of the same teaching task
  "metadata": {{
    "grade": "10",
    "subject": "Physics",
    "topic": "Newton's Laws",
    "difficulty": "medium"
  }}
}}

The possible `type` values are:
- "concept" for the main idea or explanation
- "solution" for step-by-step problem-solving
- "qa" for a Q&A pair (use both question and answer in `text`)

If multiple Q&A pairs are found, create one dictionary per pair with `type = "qa"` and format the text as:

"Q: What is inertia?\\nA: Inertia is the resistance of..."

You MUST structure all the given data into json. Do not miss any sentence. By default, the text is a concept.

---

### Example Input

Text:
\"\"\"
Grade: 10
Subject: Physics
Topic: Newton's Laws
Difficulty: Medium

Newton’s First Law states that an object at rest stays at rest and an object in motion stays in motion unless acted upon by a net external force.

To solve such problems, apply Newton’s First Law. If no net force is applied, the object maintains its velocity.

Q: What is inertia?
A: Inertia is the tendency of an object to resist changes in its motion.

Q: Why does the object remain at rest?
A: Because no net external force is acting on it.
\"\"\"

---

### Example Output

[
  {{
    "id": "{doc_id}_concept",
    "text": "Newton’s First Law states that an object at rest stays at rest and an object in motion stays in motion unless acted upon by a net external force.",
    "type": "concept",
    "doc_id": "{doc_id}",
    "metadata": {{
      "grade": "10",
      "subject": "Physics",
      "topic": "Newton's Laws",
      "difficulty": "medium"
    }}
  }},
  {{
    "id": "{doc_id}_solution",
    "text": "To solve such problems, apply Newton’s First Law. If no net force is applied, the object maintains its velocity.",
    "type": "solution",
    "doc_id": "{doc_id}",
    "metadata": {{
      "grade": "10",
      "subject": "Physics",
      "topic": "Newton's Laws",
      "difficulty": "medium"
    }}
  }},
  {{
    "id": "{doc_id}_qa_1",
    "text": "Q: What is inertia?\\nA: Inertia is the tendency of an object to resist changes in its motion.",
    "type": "qa",
    "doc_id": "{doc_id}",
    "metadata": {{
      "grade": "10",
      "subject": "Physics",
      "topic": "Newton's Laws",
      "difficulty": "medium"
    }}
  }},
  {{
    "id": "{doc_id}_qa_2",
    "text": "Q: Why does the object remain at rest?\\nA: Because no net external force is acting on it.",
    "type": "qa",
    "doc_id": "{doc_id}",
    "metadata": {{
      "grade": "10",
      "subject": "Physics",
      "topic": "Newton's Laws",
      "difficulty": "medium"
    }}
  }}
]

---

Now, extract and format the given teaching task accordingly. Return only the JSON list.

Teaching Task:
\"\"\"{doc_text}\"\"\"
"""

def get_chunck_splitter_prompt():
   return """
You are an expert educational AI assistant. Your job is to deeply analyze the provided assignment text and extract, expand, and generate all necessary semantic chunks to fully support a student through every phase of the learning process, even if the original material is incomplete or lacks detail.

For each assignment, you MUST:
- Extract all relevant information for each chunk type below.
- If any type is missing, INFER and GENERATE high-quality, pedagogically sound content based on your expertise and the context of the assignment.
- Even if a chunk type already exists in the assignment, you may and should generate additional, relevant content for that type if it will help the student progress through the learning steps. Do not limit yourself to only one chunk per type.
- Ensure that the output is always sufficient for a student to progress through all the following learning steps: "orientation", "conceptualization", "solution ideation", "planning", and "execution support".
- Be proactive: add clarifications, examples, instructions, and explanations as needed so the student never faces a dead end or lack of guidance at any step.
- Use clear, structured, and student-friendly language.

Chunk types (use ALL if possible):
- "concept": The main idea or core definition of the assignment. If not explicitly stated, infer and formulate it yourself.
- "solution": Solutions, hints, or problem-solving strategies relevant to the assignment. If not present, suggest possible approaches based on the content.
- "qa": All questions that the student is expected to answer. Identify both explicit and implicit questions. If missing, generate relevant questions and answers.
- "example": Illustrative examples that clarify the assignment. If none are provided, create a suitable example.
- "definition": Clear term-definition pairs. Extract or generate these as needed.
- "instruction": Specific tasks or steps the student must perform. Identify all actionable instructions. If the assignment is vague, break it down into concrete steps.
- "table": Any tables of data present in the assignment. Structure them clearly; if none exist, do not fabricate.

Guidelines:
- Each chunk should be at least 100 words long, if possible. UNLESS it is a table. One table MUST be whole in one chunk.
- Tables are important. Include all values from tables in one chunck
- Do not omit any sentence from the assignment text; ensure all content is included in at least one chunk.
- Use ONLY these types: concept, solution, qa, example, definition, instruction, table
- If a required chunk type is missing from the assignment, generate it based on your analysis and understanding.
- The generated chunks will be stored as vectors and used in a Retrieval-Augmented Generation (RAG) system to support the following educational steps: "orientation", "conceptualization", "solution ideation", "planning", and "execution support". Structure and formulate each chunk so that it can be effectively used for these phases, ensuring clarity, completeness, and pedagogical value for each step.
- Return ONLY a list of JSON objects, one per chunk, in the following format:
[
  {"type": "concept", "text": "..."},
  {"type": "qa", "text": "..."},
  ...
]
- Be thorough and ensure the output is as complete and helpful as possible for downstream educational applications. If the original material is sparse, supplement it with your own expert knowledge to ensure the student is never left without guidance.
"""

def get_rag_query_prompt(history, query):
    return f"""
You are an AI assistant that resolve references.

Given the conversation history and the latest user question, your only task is to resolve references.

- Use the history ONLY to resolve references (such as "this", "it", "the task").
- Query MUST remain the same but with resolved references.
- If query doesn't have references, remain it exactly the same

---

Conversation history:
{history}

Latest user question:
{query}

---

Example 1:
Query: How to use it?
History: What is RAG
Output: How to use RAG

Example 2:
Query: What is it?
History: Usage of RAG
Output: Definition of RAG

---

Return ONLY the reformulated query. Do not add any explanations or extra content.
"""

def get_student_simulation_prompt(history, last_response):
    return f"""
You are simulating a motivated, curious student interacting with an AI teaching assistant. Your goal is to learn as much as possible, clarify any doubts, and make steady progress through the assignment.

- Read the assistant's last response carefully.
- If you understand, ask a relevant follow-up question, request clarification, examples or try to apply what you learned.
- If something is unclear, ask for an explanation or an example.
- If you feel ready, ask about the next step or how to proceed.
- Be specific and natural in your questions, as a real student would.
- Do not invent information about the assignment; base your questions only on what you have seen so far.
- If you feel lost, say so and ask for help.
- If you have completed the assignment, thank the assistant and say you are done.

---

Conversation history:
{history}

Assistant's last response:
{last_response}

---

Return ONLY the next message from the student, as if you are the student continuing the conversation. Do not add any explanations or extra content.
"""