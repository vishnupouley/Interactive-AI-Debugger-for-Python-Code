import gradio as gr
from groq import Groq, GroqError
import os
import json # Not strictly needed for this version, but good for future complex metadata

# --- Configuration ---
# IMPORTANT: Set your Groq API key as an environment variable before running.
# **CHANGE THIS TO YOUR DESIRED AND AVAILABLE MODEL.**
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct" # Or "llama-4-scout-17b-16e-instruct" if available
# --- LLM Interaction Function ---
def get_llm_code_explanation(user_message: str, chat_history: list[list[str | None]]):
    """
    Sends the user's message and chat history to the Groq LLM
    and returns the AI's response.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        # This message will be displayed in the chat if the API key is not set.
        return "Error: GROQ_API_KEY environment variable not set. Please set it and restart the application."
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
    # Construct the conversation history in the format expected by the API
    messages_for_api = [{"role": "system", "content": system_prompt_content}]
    for user_msg_in_history, ai_msg_in_history in chat_history:
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
            stream=True, # Set to True if you want streaming responses
            # stop=None, # Optional: sequences where the API will stop generating further tokens
        )
        response_content = chat_completion.choices[0].message.content
        return response_content
    except GroqError as e:
        error_message = f"Groq API Error: {e.message or str(e)}"
        # Check for common issues like invalid API key or model not found
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            error_message += "\nPlease check if your GROQ_API_KEY is correct and has permissions."
        if "model_not_found" in str(e).lower() or "not found" in str(e).lower() and GROQ_MODEL_NAME in str(e).lower():
            error_message += f"\nThe model '{GROQ_MODEL_NAME}' might not be available. Please check the model name and your Groq account."
        return error_message
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
# --- Gradio Interface Definition ---
# Using gr.ChatInterface which bundles Chatbot display and Textbox input
# type="messages" ensures history is passed as a list of [user, AI] message pairs.
# autofocus=False as per your example.
python_code_debugger_chat = gr.ChatInterface(
    fn=get_llm_code_explanation,
    title="Python Code Debugger & Explainer AI",
    description=(
        f"Enter your Python code, optionally with an error message or question. "
        f"The AI (using Groq with model: {GROQ_MODEL_NAME}) will help you debug and understand it.\n"
    ),
    examples=[
        ["def my_func()\n  print('Hello')\nmy_func()", "IndentationError: expected an indented block"],
        ["data = {'key': 'value'}\nprint(data['non_existent_key'])"],
        ["numbers = [1, 2, 3]\nfor i in range(4):\n  print(numbers[i])", "Help me fix the IndexError."],
        ["# This is my Python script to sort a list\nitems = [5, 1, 9, 3]\n# How can I sort this in descending order?"]
    ],
    type="messages",
    autofocus=False,
    theme="soft", # You can try other themes like "default", "huggingface", "gradio/glass"
    chatbot=gr.Chatbot(
        label="Code Debugging Assistant",
        show_label=True,
        bubble_full_width=False, # Makes chat bubbles not take full width
        avatar_images=(
            "https://img.icons8.com/ios-glyphs/30/000000/user--v1.png", # User avatar (placeholder)
            "https://img.icons8.com/fluency/48/bot.png"  # Bot avatar (placeholder)
        ),
        height=600 # Set a height for the chatbot display area
    ),
    textbox=gr.Textbox(
        placeholder="Paste your Python code here, describe the error, or ask a question...",
        lines=4 # Start with a slightly larger textbox
    )
)
# --- Launch the Application ---
if __name__ == "__main__":
    print("Starting the Python Code Debugger App...")
    print(f"Using Groq model: {GROQ_MODEL_NAME}")
    # share=True would create a public link if you want to share it (requires internet)
    # server_name="0.0.0.0" makes it accessible on your local network
    python_code_debugger_chat.launch(server_name="0.0.0.0", server_port=7871)
