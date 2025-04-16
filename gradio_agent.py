import os
import asyncio
import logging
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerHTTP
from dotenv import load_dotenv

# --- Logging Setup ---
# It's generally good practice to set up logging once,
# possibly at the application entry point if used elsewhere.
# If this class is the main component, setting it up here is fine.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
# Load .env file. Consider if this should happen outside the class
# if multiple components need environment variables.
load_dotenv(override=True)
logger.info("Environment variables loading attempt finished.")


class GradioAgent:
    """
    A class to encapsulate the Pydantic AI Agent setup and execution
    for use within a Gradio interface.
    """

    def __init__(self):
        """
        Initializes the agent, including LLM provider, model, and MCP server.
        """
        logger.info("Initializing GradioAgent...")

        # --- Configuration ---
        llm_api_key = os.getenv("API_KEY", "OLLAMA")
        llm_endpoint = os.getenv("BASE_URL", "http://localhost:11434/v1")
        llm_model_name = os.getenv("MODEL_NAME", "qwen2.5:latest")
        mcp_server_host = os.getenv("SERVER_HOST", "127.0.0.1")  # Default added
        mcp_server_port = int(os.getenv("SERVER_PORT", "8080"))  # Default added

        logger.info(f"Using LLM Endpoint: {llm_endpoint}")
        logger.info(f"Using LLM Model: {llm_model_name}")
        logger.info(f"MCP Server Host: {mcp_server_host}")
        logger.info(f"MCP Server Port: {mcp_server_port}")

        if not all([llm_endpoint, llm_model_name, mcp_server_host, mcp_server_port]):
            logger.warning(
                "One or more environment variables (BASE_URL, MODEL_NAME, SERVER_HOST, SERVER_PORT) might be missing."
            )
            # Depending on requirements, you might want to raise an error here
            # raise ValueError("Required environment variables are not set.")

        # --- LLM Provider and Model ---
        try:
            provider = OpenAIProvider(base_url=llm_endpoint, api_key=llm_api_key)
            model = OpenAIModel(provider=provider, model_name=llm_model_name)
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider/model: {e}", exc_info=True)
            raise  # Re-raise the exception to prevent agent initialization with bad config

        # --- MCP Server ---
        logger.info("Initializing MCP Server HTTP")
        try:
            # Ensure port is an integer
            mcp_server_url = (
                f"http://{mcp_server_host}:{mcp_server_port}/mcp"  # Added http://
            )
            self.mcp_server = MCPServerHTTP(url=mcp_server_url)
            logger.info(f"MCP Server HTTP initialized with URL: {mcp_server_url}")
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}", exc_info=True)
            raise  # Re-raise

        # --- Agent ---
        logger.info("Initializing Agent...")
        try:
            # Store the agent as an instance variable
            self.agent = Agent(model, mcp_servers=[self.mcp_server], retries=3)
            logger.info("Agent initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Agent: {e}", exc_info=True)
            raise  # Re-raise

    async def run_inference(self, prompt: str) -> str:
        """
        Runs the agent with the provided prompt.

        Args:
            prompt: The input prompt for the agent.

        Returns:
            The result string from the agent.

        Raises:
            Exception: If the agent fails to run after retries.
        """
        if not prompt:
            logger.warning("Received empty prompt.")
            return "Please provide a valid prompt."

        logger.info(f"Running agent with prompt: {prompt.strip()}")
        try:
            # The MCP server needs to be running during the agent execution
            async with self.agent.run_mcp_servers():
                logger.info("MCP Server context entered.")
                result = await self.agent.run(prompt)
                logger.info("MCP Server context exited.")

            logger.info(f"Agent finished. Result length: {len(result)}")
            # logger.debug(f"Agent result: {result}") # Use debug level for potentially large output
            return result
        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            # Depending on Gradio needs, you might return an error message
            # or re-raise the exception.
            return f"An error occurred: {e}"


# --- Example Usage (Optional - for testing the class directly) ---
async def _test_agent_class():
    print("Testing GradioAgent class...")
    try:
        gradio_agent = GradioAgent()
        test_prompt = "Was bedeutet KI? Nenne 5 Quellen."
        # test_prompt = "What is the capital of France?" # English example
        result = await gradio_agent.run_inference(test_prompt)
        print("\n--- Agent Result ---")
        print(result)
        print("--- End Agent Result ---")
    except Exception as e:
        print(f"\n--- Error during testing ---")
        print(f"An error occurred: {e}")
        # You might want more detailed logging or error handling here
        import traceback

        traceback.print_exc()
        print("--- End Error ---")


if __name__ == "__main__":
    print("Starting agent class test...")
    # Ensure an event loop is running if testing directly
    # asyncio.run(_test_agent_class()) # Preferred for Python 3.7+

    # Alternative for environments where asyncio.run might cause issues
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If in an environment like Jupyter where a loop is already running
            import nest_asyncio

            nest_asyncio.apply()
            loop.create_task(_test_agent_class())
        else:
            loop.run_until_complete(_test_agent_class())
    except RuntimeError:
        # Fallback if get_event_loop fails in certain contexts
        asyncio.run(_test_agent_class())

    print("Agent class test finished.")
