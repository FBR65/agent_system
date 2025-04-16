import os
import asyncio
import logging
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.mcp import MCPServerHTTP
from dotenv import load_dotenv

os.environ["SERVER_HOST"] = "127.0.0.1"
os.environ["SERVER_PORT"] = str(8504)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)
logger.info("Environment variables loaded.")

LLM_API_KEY = os.getenv("API_KEY", "OLLAMA")
LLM_ENDPOINT = os.getenv("BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("MODEL_NAME", "qwen2.5:latest")

logger.info(f"Using LLM Endpoint: {LLM_ENDPOINT}")
logger.info(f"Using LLM Model: {LLM_MODEL}")

provider = OpenAIProvider(base_url=LLM_ENDPOINT, api_key=LLM_API_KEY)

model = OpenAIModel(provider=provider, model_name=LLM_MODEL)

# Define the MCP Servers
logger.info("Initializing MCP Server HTTP")
mcp_server_url = (
    f"http://{os.environ['SERVER_HOST']}:{int(os.environ['SERVER_PORT'])}/mcp"
)
# Instantiate the MCPServerHTTP object
mcp_server = MCPServerHTTP(url=mcp_server_url)
logger.info("MCP Server HTTP initialized.")

# Define the Agent with both MCP servers
logger.info("Initializing Agent...")
agent = Agent(model, mcp_servers=[mcp_server], retries=3)
logger.info("Agent initialized.")


# Main async function
async def main():
    logger.info("Starting Agent")
    async with agent.run_mcp_servers():
        prompt = """
        Was bedeutet KI? Nenne 5 Quellen.
        """
        logger.info(f"Running agent with prompt: {prompt.strip()}")
        result = await agent.run(prompt)
        logger.info(f"Agent finished. Result: {result}")
        print(result)
    logger.info("Agent stopped.")


# Run the async function
if __name__ == "__main__":
    logger.info("Starting application.")
    asyncio.run(main())
    logger.info("Application finished.")
