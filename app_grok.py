import gradio as gr
from groq import Client

# Set your API key (replace "your_api_key_here" with your actual Groq API key)
api_key = "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"
client = Client(api_key=api_key)

# Initialize global history with a system message
global_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

def respond(message):
    global global_history
    # Add user message to global history
    global_history.append({"role": "user", "content": message})
    
    # Call Groq API with the full conversation history
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=global_history,
        max_tokens=100,  # Adjust as needed
        temperature=0.7,  # Adjust as needed
    )
    
    # Extract and add assistant's response to global history
    bot_message = response.choices[0].message.content
    global_history.append({"role": "assistant", "content": bot_message})
    
    # Convert global history to Gradio format (skip system message)
    gradio_history = []
    for i in range(1, len(global_history), 2):
        user_msg = global_history[i]["content"]
        if i + 1 < len(global_history):
            bot_msg = global_history[i + 1]["content"]
        else:
            bot_msg = None  # This should not happen in normal operation
        gradio_history.append([user_msg, bot_msg])
    
    return gradio_history

# Set up Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message here")
    
    # Submit button functionality
    msg.submit(respond, msg, chatbot)
    msg.submit(lambda: "", None, msg)  # Clear the input textbox after submission

# Launch the demo
demo.launch()