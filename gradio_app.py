import gradio as gr
import logging
from gradio_agent import GradioAgent

# --- Logging Setup (Optional but recommended for Gradio app) ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Agent Initialization ---
agent_instance = None
agent_ready = False
initialization_error_message = ""

try:
    logger.info("Initializing GradioAgent...")
    agent_instance = GradioAgent()
    agent_ready = True
    logger.info("GradioAgent initialized successfully.")
except Exception as e:
    error_msg = f"FATAL: Failed to initialize GradioAgent: {e}"
    logger.error(error_msg, exc_info=True)  # Log with traceback
    initialization_error_message = (
        f"Error: Agent could not be initialized. Please check server logs. Details: {e}"
    )
    # agent_instance remains None
    # agent_ready remains False


# --- Gradio Interaction Logic ---
async def handle_chat(prompt: str) -> str:
    """Handles the chat interaction, calling the agent."""
    if not agent_ready or agent_instance is None:
        logger.warning("Attempted to run chat interface, but agent is not ready.")
        # Return the specific error captured during init
        return initialization_error_message

    if not prompt or not prompt.strip():
        logger.warning("Received empty prompt.")
        return "Please enter a prompt."

    logger.info(f"Received prompt: '{prompt[:50]}...'")  # Log snippet
    try:
        # Call the async method of your agent instance
        result = await agent_instance.run_inference(prompt)
        logger.info("Agent inference successful.")
        return result
    except Exception as e:
        logger.error(f"Error during agent inference: {e}", exc_info=True)
        return f"An error occurred while processing your request: {e}"


# --- Gradio UI Definition using Blocks ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # AI Agent Interface
        Ask the AI agent anything. It can use tools via the MCP server.
        """
    )
    with gr.Row():
        prompt_input = gr.Textbox(
            label="Your Prompt",
            placeholder="Enter your prompt here...",
            lines=3,
            scale=4,  # Make input wider
        )
        submit_button = gr.Button("Run Agent", variant="primary", scale=1)

    output_display = gr.Textbox(
        label="Agent Response",
        lines=10,
        interactive=False,  # Output is not editable by user
    )

    # --- Connect UI elements to logic ---
    submit_button.click(
        fn=handle_chat,  # The async function to call
        inputs=prompt_input,  # Input component(s)
        outputs=output_display,  # Output component(s)
    )

    # Optional: Allow submitting by pressing Enter in the textbox
    prompt_input.submit(fn=handle_chat, inputs=prompt_input, outputs=output_display)

    # Display initialization error prominently if agent failed to load
    if not agent_ready:
        gr.Markdown(f"<h3 style='color:red;'>{initialization_error_message}</h3>")


# --- Launch the Gradio app ---
if __name__ == "__main__":
    if agent_ready:
        logger.info("Launching Gradio interface...")
        # You might want to add share=True for public access if needed
        demo.launch(server_name="0.0.0.0")
    else:
        logger.error(
            "Gradio interface cannot be launched due to agent initialization failure."
        )
        # Keep the app running to show the error message in the UI
        print("Agent initialization failed. Launching Gradio interface to show error.")
        demo.launch(server_name="0.0.0.0")
