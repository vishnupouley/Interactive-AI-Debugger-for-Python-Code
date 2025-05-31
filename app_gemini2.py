import gradio as gr
from groq import Groq, GroqError
import os
import time

# --- Configuration ---
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct" # CHANGE if needed
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
"""

# --- Global State ---
# WARNING: This global variable stores ONE chat history for ALL users.
# It persists as long as the Python process runs. It survives refreshes.
# DO NOT use this simple approach for multi-user production apps.
global_chat_history = [] # List of [user_message, ai_message] pairs

# --- LLM Interaction Function ---
def call_groq_llm(history_to_process):
    """Calls Groq LLM with the provided history and returns the response."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY environment variable not set."

    client = Groq(api_key=api_key)

    messages_for_api = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, ai_msg in history_to_process:
        if user_msg:
            messages_for_api.append({"role": "user", "content": user_msg})
        if ai_msg: # Only add AI msg if it exists (it won't for the last user msg)
            messages_for_api.append({"role": "assistant", "content": ai_msg})

    try:
        chat_completion = client.chat.completions.create(
            messages=messages_for_api,
            model=GROQ_MODEL_NAME,
            temperature=0.7,
            max_tokens=3500,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except GroqError as e:
        return f"Groq API Error: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# --- Gradio Interface Definition using gr.Blocks ---
with gr.Blocks(theme="soft", title="Python Debugger (Persistent)") as demo:
    gr.Markdown(
        f"""
        # Python Code Debugger & Explainer (Persistent Global History)
        **Warning:** This version uses a single global history on the server.
        It will persist through refreshes but will be shared if multiple users access it.
        Using Groq model: {GROQ_MODEL_NAME}
        """
    )
    
    chatbot = gr.Chatbot(label="Chat History", height=600, bubble_full_width=False)
    msg_box = gr.Textbox(
        placeholder="Paste your Python code here, describe the error, or ask a question...",
        lines=4,
        show_label=False,
        autofocus=True,
    )
    clear_button = gr.Button("Clear Chat History")

    def handle_user_submission(user_message):
        """
        Handles user input: updates global history, calls LLM, and yields updates.
        """
        global global_chat_history

        # 1. Add user message to global history and yield to show it
        global_chat_history.append([user_message, None])
        yield "", global_chat_history

        # 2. Call LLM with a copy of the current history
        bot_response = call_groq_llm(list(global_chat_history)) # Pass a copy

        # 3. Update the last entry in global history with the response
        global_chat_history[-1][1] = bot_response

        # 4. Yield the final state
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
    # When the user submits text (presses Enter or clicks a submit button if we had one)
    msg_box.submit(handle_user_submission, [msg_box], [msg_box, chatbot])

    # When the Gradio app loads in a browser
    demo.load(load_history_on_start, inputs=None, outputs=[chatbot])
    
    # When the clear button is clicked
    clear_button.click(clear_chat, inputs=None, outputs=[chatbot])

# --- Launch the Application ---
if __name__ == "__main__":
    print("Starting the Python Code Debugger App (Global History Mode)...")
    print("WARNING: This mode uses a single chat history shared by all users.")
    print(f"Using Groq model: {GROQ_MODEL_NAME}")
    
    # You can run this with `python app.py` and it will use Gradio's server.
    # To run with uvicorn, you'd typically mount this 'demo' object onto
    # a FastAPI app, but running directly is simpler for this example.
    demo.launch(server_name="0.0.0.0", server_port=7872)

""" ```

**How This Works:**

1.  **`global_chat_history = []`**: This list exists in your Python server's memory. It's the *single source of truth*.
2.  **`gr.Blocks()`**: We define the UI manually.
3.  **`demo.load(load_history_on_start, ...)`**: Whenever *any* browser connects or refreshes, it runs `load_history_on_start`. This function simply returns whatever is currently in `global_chat_history`, and Gradio sends it to the `chatbot` component in that browser.
4.  **`msg_box.submit(handle_user_submission, ...)`**: When you send a message:
    * `handle_user_submission` is called.
    * It *modifies the `global_chat_history`*.
    * It uses `yield` to send updates back to the browser immediately (showing your message first, then the bot's response).
    * Because the `global_chat_history` is updated, the *next time* any browser loads or refreshes, it will get this updated history.

This approach effectively bypasses the browser's session state and cookie mechanisms for history storage, making it resistant to hard refreshes and cookie issues. It stores the state purely on the server, and it lasts precisely as long as your Python script is running. """