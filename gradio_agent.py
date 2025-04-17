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
        mcp_server_port = int(os.getenv("SERVER_PORT", "8504"))  # Default added

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
                # agent.run returns an AgentRunResult object
                agent_run_result = await self.agent.run(prompt)
                logger.info("MCP Server context exited.")

            # --- Debugging: Inspect the result object ---
            # You can remove these lines now
            # print("\n--- Inspecting AgentRunResult ---")
            # print(f"Type: {type(agent_run_result)}")
            # print(f"Object: {agent_run_result}")
            # print(f"Attributes: {dir(agent_run_result)}")
            # try:
            #     print(f"Vars: {vars(agent_run_result)}")
            # except TypeError:
            #     print("Vars: Not applicable for this object type.")
            # print("--- End Inspection ---")
            # --- End Debugging ---

            # Access the 'data' attribute to get the final string output
            final_output = agent_run_result.data
            logger.info("Successfully extracted 'data' from AgentRunResult.")

            # Now you can safely get the length of the output string
            logger.info(f"Agent finished. Result length: {len(final_output)}")
            # logger.debug(f"Agent result: {final_output}") # Use debug level for potentially large output

            # Return the extracted string
            return final_output
        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            # Depending on Gradio needs, you might return an error message
            # or re-raise the exception.
            return f"An error occurred: {e}"


async def _test_agent_class():
    print("Testing GradioAgent class...")
    try:
        gradio_agent = GradioAgent()
        # Ensure the LLM endpoint read is correct (optional debug print)
        # print(f"DEBUG: Provider Base URL: {gradio_agent.agent.model.provider.base_url}")
        test_prompt = "Was bedeutet KI? Nenne 5 Quellen."
        # test_prompt = "What is the capital of France?" # English example
        result = await gradio_agent.run_inference(test_prompt)
        print("\n--- Agent Result ---")
        print(result)
        print("--- End Agent Result ---")
    except Exception as e:
        print(f"\n--- Error during testing ---")
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
        print("--- End Error ---")


if __name__ == "__main__":
    print("Starting agent class test...")
    # Use the standard asyncio.run for direct script execution
    try:
        asyncio.run(_test_agent_class())
    except Exception as e:
        # Catch potential top-level errors during asyncio.run itself
        print(f"\n--- Top-level Error during testing ---")
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
        print("--- End Top-level Error ---")

    print("Agent class test finished.")
