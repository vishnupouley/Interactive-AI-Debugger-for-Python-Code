from groq import Groq, GroqError
from dotenv import load_dotenv
import gradio as gr
import json # Not strictly needed for this version, but good for future complex metadata
import os

load_dotenv()

# --- Configuration ---
# IMPORTANT: Set your Groq API key as an environment variable before running.
# **CHANGE THIS TO YOUR DESIRED AND AVAILABLE MODEL.**
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct" # Or "llama-4-scout-17b-16e-instruct" if available

# --- Global State for Chat Persistence ---
# WARNING: This global variable stores ONE chat history for ALL users.
# It persists as long as the Python process runs. It survives refreshes.
# DO NOT use this simple approach for multi-user production apps.
global_chat_history = [] # List of [user_message, ai_message] pairs

# --- LLM Interaction Function ---
def get_llm_code_explanation(user_message: str, chat_history: list[list[str | None]]):
    """
    Sends the user's message and chat history to the Groq LLM
    and returns the AI's response. Uses global history for persistence.
    """
    global global_chat_history
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        # This message will be displayed in the chat if the API key is not set.
        response = "Error: GROQ_API_KEY environment variable not set. Please set it and restart the application."
        global_chat_history.append([user_message, response])
        return response
    
    client = Groq(api_key=api_key)
    
    # System prompt to guide the LLM
    system_prompt_content = """You are an expert Python Code Debugging Assistant.
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
    
    # Use global chat history for building the conversation context
    messages_for_api = [{"role": "system", "content": system_prompt_content}]
    for user_msg_in_history, ai_msg_in_history in global_chat_history:
        if user_msg_in_history: # Add user message from history
            messages_for_api.append({"role": "user", "content": user_msg_in_history})
        if ai_msg_in_history: # Add assistant message from history
            messages_for_api.append({"role": "assistant", "content": ai_msg_in_history})
    
    # Add the current user message
    messages_for_api.append({"role": "user", "content": user_message})
    
    try:
        chat_completion = client.chat.completions.create(
            messages=messages_for_api,
            model=GROQ_MODEL_NAME,
            temperature=0.7,  # Adjust for creativity vs. factuality
            max_tokens=3500, # Increased for potentially long code snippets and detailed explanations
            top_p=1,
            stream=False, # Set to False for simpler handling
            # stop=None, # Optional: sequences where the API will stop generating further tokens
        )
        response_content = chat_completion.choices[0].message.content
        
        # Update global chat history
        global_chat_history.append([user_message, response_content])
        
        return response_content
    except GroqError as e:
        error_message = f"Groq API Error: {e.message or str(e)}"
        # Check for common issues like invalid API key or model not found
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            error_message += "\nPlease check if your GROQ_API_KEY is correct and has permissions."
        if "model_not_found" in str(e).lower() or "not found" in str(e).lower() and GROQ_MODEL_NAME in str(e).lower():
            error_message += f"\nThe model '{GROQ_MODEL_NAME}' might not be available. Please check the model name and your Groq account."
        
        # Still update global history even for errors
        global_chat_history.append([user_message, error_message])
        return error_message
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        global_chat_history.append([user_message, error_message])
        return error_message

# Custom function to handle chat interface with persistence
def chat_fn(message, history):
    """
    Custom chat function that integrates with global history.
    This function is called by gr.ChatInterface.
    """
    global global_chat_history
    
    # If this is the first message or global history is empty, sync with current history
    if not global_chat_history and history:
        global_chat_history = history.copy()
    
    # Get response from LLM
    response = get_llm_code_explanation(message, history)
    
    # Return the response - gr.ChatInterface will handle adding it to the display
    return response

# --- Create Gradio Interface with Blocks for better control ---
with gr.Blocks(theme="soft", title="Interactive AI Debugger for Python Code") as demo:
    gr.Markdown(
        f"""
        # Python Code Debugger & Explainer AI
        Enter your Python code, optionally with an error message or question. 
        The AI (with model: {GROQ_MODEL_NAME}) will help you debug and understand it.
        """
    )
    
    # Create the chat interface
    chatbot = gr.Chatbot(
        label="Code Debugging Assistant",
        show_label=True,
        height=600,
        type="messages"  # Fix the deprecation warning
    )
    
    msg_box = gr.Textbox(
        placeholder="Paste your Python code here, describe the error, or ask a question...",
        lines=3,
        show_label=False,
        autofocus=False
    )
    
    clear_btn = gr.Button("Clear Chat History", variant="secondary")
    
    # Example buttons
    with gr.Row():
        example1 = gr.Button("Example: Syntax Error", size="sm")
        example2 = gr.Button("Example: KeyError", size="sm")
        example3 = gr.Button("Example: IndexError", size="sm")
        example4 = gr.Button("Example: Sorting Question", size="sm")
    
    def respond(message, chat_history):
        """Handle user message and return updated chat history"""
        global global_chat_history
        
        # Get bot response
        bot_message = get_llm_code_explanation(message, chat_history)
        
        # Update chat history
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        
        return "", chat_history
    
    def load_history():
        """Load existing global history when page loads"""
        global global_chat_history
        # Convert to messages format
        messages = []
        for user_msg, ai_msg in global_chat_history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if ai_msg:
                messages.append({"role": "assistant", "content": ai_msg})
        return messages
    
    def clear_history():
        """Clear the global chat history"""
        global global_chat_history
        global_chat_history = []
        return []
    
    def set_example(example_text):
        """Set example text in the message box"""
        return example_text
    
    # Event handlers
    msg_box.submit(respond, [msg_box, chatbot], [msg_box, chatbot])
    clear_btn.click(clear_history, inputs=None, outputs=[chatbot])
    
    # Example button handlers
    example1.click(
        lambda: "def my_func()\n  print('Hello')\nmy_func()\n# IndentationError: expected an indented block",
        outputs=[msg_box]
    )
    example2.click(
        lambda: "data = {'key': 'value'}\nprint(data['non_existent_key'])",
        outputs=[msg_box]
    )
    example3.click(
        lambda: "numbers = [1, 2, 3]\nfor i in range(4):\n  print(numbers[i])\n# Help me fix the IndexError.",
        outputs=[msg_box]
    )
    example4.click(
        lambda: "# This is my Python script to sort a list\nitems = [5, 1, 9, 3]\n# How can I sort this in descending order?",
        outputs=[msg_box]
    )
    
    # Load existing history when the page loads
    demo.load(load_history, inputs=None, outputs=[chatbot])

# --- Launch the Application ---
if __name__ == "__main__":
    print("Starting the Python Code Debugger App with Persistent Chat...")
    print("WARNING: Chat history persists across refreshes and is shared by all users.")
    print(f"Using Groq model: {GROQ_MODEL_NAME}")
    
    # share=True would create a public link if you want to share it (requires internet)
    # server_name="0.0.0.0" makes it accessible on your local network
    demo.launch(server_name="0.0.0.0", server_port=7874)