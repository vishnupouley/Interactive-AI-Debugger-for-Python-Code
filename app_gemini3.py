import gradio as gr
from groq import Groq, GroqError
import os

# --- Configuration ---
# IMPORTANT: Set your Groq API key as an environment variable before running.
# **CHANGE THIS TO YOUR DESIRED AND AVAILABLE MODEL.**
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct" # User specified model
SYSTEM_PROMPT = """You are an expert Python Code Debugging Assistant.
A user will provide Python code, and optionally an error message or a description of a problem.
Your primary tasks are to:
1.  Carefully analyze the provided Python code.
2.  If an error message is given, focus on explaining that specific error.
3.  If no error message is given, try to identify potential bugs, logical errors, or areas for improvement in the code.
4.  Explain any identified errors or issues in a clear, concise, and step-by-step manner.
5.  Provide the corrected or improved Python code.
6.  Ensure the corrected code is well-formatted. Use Markdown for Python code blocks, like this:
    ```python
    # your corrected code here
    print("Hello, World!")
    ```
7.  If the query is not about Python code or doesn't contain code, respond as a helpful general assistant.
8.  Be friendly and encouraging.
"""

# --- Global State ---
# WARNING: This global variable stores ONE chat history for ALL users.
# It persists as long as the Python process runs. It survives refreshes.
# DO NOT use this simple approach for multi-user production apps.
global_chat_history = [] # List of [user_message, ai_message] pairs

# --- LLM Interaction Function (Streaming) ---
def call_groq_llm_stream(history_to_process):
    """
    Calls Groq LLM with the provided history and yields response chunks.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        yield "Error: GROQ_API_KEY environment variable not set. Please set it and restart the application."
        return

    client = Groq(api_key=api_key)

    messages_for_api = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Construct messages from history_to_process, which is global_chat_history
    # It already includes the latest user message with AI part as None or ""
    for user_msg, ai_msg in history_to_process:
        if user_msg:
            messages_for_api.append({"role": "user", "content": user_msg})
        if ai_msg and ai_msg != "": # Add AI message if it's part of a completed turn
             # This condition is tricky for the last turn if AI response is being built.
             # For the API call, the last AI message should not be sent if it's the one we are generating.
             # The history_to_process passed to this function should represent the state *before* this AI turn.
             # Let's adjust how history_to_process is formed or handled.
             # For now, assuming history_to_process is correctly formatted for the API call.
             # The current `handle_user_submission` sends the full history including the user's latest message.
             pass # The API expects the last message to be from the user if we are generating a response.

    # More robust message construction for API:
    api_messages_for_llm = [{"role": "system", "content": SYSTEM_PROMPT}]
    for hist_user_msg, hist_ai_msg in history_to_process[:-1]: # All turns except the current one
        if hist_user_msg: api_messages_for_llm.append({"role": "user", "content": hist_user_msg})
        if hist_ai_msg: api_messages_for_llm.append({"role": "assistant", "content": hist_ai_msg})
    # Add the current user's message for which we need a response
    if history_to_process and history_to_process[-1][0]:
        api_messages_for_llm.append({"role": "user", "content": history_to_process[-1][0]})


    try:
        stream = client.chat.completions.create(
            messages=api_messages_for_llm, # Use the carefully constructed messages
            model=GROQ_MODEL_NAME,
            temperature=0.7,
            max_tokens=3500,
            top_p=1,
            stream=True,
        )
        for chunk in stream:
            content_chunk = chunk.choices[0].delta.content
            if content_chunk:
                yield content_chunk
    except GroqError as e:
        error_message = f"Groq API Error: {e.message or str(e)}"
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            error_message += "\nPlease check if your GROQ_API_KEY is correct and has permissions."
        if "model_not_found" in str(e).lower() or "not found" in str(e).lower() and GROQ_MODEL_NAME in str(e).lower():
            error_message += f"\nThe model '{GROQ_MODEL_NAME}' might not be available. Please check the model name and your Groq account."
        yield error_message
    except Exception as e:
        yield f"An unexpected error occurred: {str(e)}"

# --- Gradio Interface Definition using gr.Blocks ---
with gr.Blocks(theme="soft", title="Python Debugger (Persistent & Streaming)") as demo:
    gr.Markdown(
        f"""
        # Python Code Debugger & Explainer AI
        Enter your Python code, optionally with an error message or question.
        The AI (using Groq with model: **{GROQ_MODEL_NAME}**) will help you debug and understand it.
        
        **Warning:** This version uses a single global history on the server.
        It will persist through refreshes but will be shared if multiple users access it.

        **Examples:**
        * `def my_func()\n  print('Hello')\nmy_func()` (with error: `IndentationError: expected an indented block`)
        * `data = {{'key': 'value'}}\nprint(data['non_existent_key'])`
        * `numbers = [1, 2, 3]\nfor i in range(4):\n  print(numbers[i])` (with query: `Help me fix the IndexError.`)
        * `# This is my Python script to sort a list\nitems = [5, 1, 9, 3]\n# How can I sort this in descending order?`
        """
    )
    
    chatbot = gr.Chatbot(
        label="Code Debugging Assistant",
        height=600,
        bubble_full_width=False,
        avatar_images=(
            "https://img.icons8.com/ios-glyphs/30/000000/user--v1.png", # User avatar (placeholder)
            "https://img.icons8.com/fluency/48/bot.png"  # Bot avatar (placeholder)
        ),
        show_label=True # Matches ChatInterface example
    )
    msg_box = gr.Textbox(
        placeholder="Paste your Python code here, describe the error, or ask a question...",
        lines=4,
        show_label=False,
        autofocus=True,
    )
    clear_button = gr.Button("Clear Chat History")

    def handle_user_submission(user_message):
        """
        Handles user input: updates global history, calls LLM, and yields updates for streaming.
        """
        global global_chat_history

        if not user_message.strip(): # Do nothing if message is empty
            yield "", global_chat_history # Return current state
            return

        # 1. Add user message to global history (AI response part is initially empty)
        global_chat_history.append([user_message, ""])
        yield "", global_chat_history # Update chatbot to show user message immediately

        # 2. Stream LLM response
        # Pass the current state of global_chat_history to the LLM call
        # The LLM function will use this to construct the API messages
        ai_response_accumulator = ""
        for chunk in call_groq_llm_stream(list(global_chat_history)): # Pass a copy
            if chunk:
                ai_response_accumulator += chunk
                global_chat_history[-1][1] = ai_response_accumulator # Update the AI part of the last message
                yield "", global_chat_history # Update chatbot with new chunk
            else: # Handle potential empty chunks if API sends them, though Groq usually doesn't for content
                pass
        
        # Final update just in case (though yield in loop should cover it)
        yield "", global_chat_history


    def load_history_on_start():
        """
        Called when a browser connects/refreshes. Returns the current global history.
        """
        return global_chat_history

    def clear_chat():
        """
        Clears the global chat history and updates the chatbot.
        """
        global global_chat_history
        global_chat_history = []
        return [] # Return empty list to clear chatbot

    # Event Handlers
    msg_box.submit(handle_user_submission, [msg_box], [msg_box, chatbot])
    demo.load(load_history_on_start, inputs=None, outputs=[chatbot])
    clear_button.click(clear_chat, inputs=None, outputs=[chatbot])

# --- Launch the Application ---
if __name__ == "__main__":
    print("Starting the Python Code Debugger App (Global History & Streaming Mode)...")
    print("WARNING: This mode uses a single chat history shared by all users.")
    print(f"Using Groq model: {GROQ_MODEL_NAME}")
    print("Ensure your GROQ_API_KEY environment variable is set.")
    
    demo.launch(server_name="0.0.0.0", server_port=7870) # Updated port
