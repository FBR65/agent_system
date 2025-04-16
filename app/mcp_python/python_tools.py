import io
import base64
import matplotlib.pyplot as plt
import sys
from io import StringIO
import traceback
import logging
import os  # <-- Add os import
import datetime  # <-- Add datetime import

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"--- Starting execution of {__name__} ---")
# --- End Logging Configuration ---

# --- Output Directory for Generated Code ---
GENERATED_CODE_DIR = "generated_code"
os.makedirs(GENERATED_CODE_DIR, exist_ok=True)  # Ensure the directory exists
logger.info(f"Generated code will be saved to: {GENERATED_CODE_DIR}")
# --- End Output Directory ---


class PythonREPL:
    def run(self, code):
        logger.info(f"Executing Python code in REPL:\n```python\n{code}\n```")
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        try:
            # Use a separate dict for globals to avoid polluting the main script's globals
            exec_globals = {}
            exec(code, exec_globals)
            sys.stdout = old_stdout
            result = redirected_output.getvalue()
            logger.info(f"REPL execution successful. Output:\n{result}")
            return result
        except Exception as e:
            sys.stdout = old_stdout
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"REPL execution failed. Error:\n{error_msg}")
            return error_msg


async def data_visualization(code: str, repl: PythonREPL) -> str:
    """Execute Python code. Use matplotlib for visualization. Saves the code."""
    logger.info(f"Tool 'data_visualization' called with code:\n```python\n{code}\n```")

    # --- Save the generated code ---
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(
            GENERATED_CODE_DIR, f"visualization_code_{timestamp}.py"
        )
        with open(filename, "w", encoding="utf-8") as f:
            # Add a header comment for context (optional)
            f.write(f"# Generated code for data visualization at {timestamp}\n")
            f.write("# Tool: data_visualization\n\n")
            f.write(code)
        logger.info(f"Generated code saved to: {filename}")
    except Exception as e:
        # Log error but continue execution if possible
        logger.error(f"Failed to save generated code to {filename}: {e}")
    # --- End save code ---

    try:
        # Execute the code first to generate the plot
        logger.debug("Executing code for data visualization...")
        # Use the REPL to execute the code
        execution_output = repl.run(code)
        # Check if the execution itself returned an error message from the REPL
        if execution_output and execution_output.strip().startswith("Error:"):
            logger.error(f"Code execution failed within REPL: {execution_output}")
            # Return the execution error instead of trying to save a plot
            return f"Code execution failed:\n{execution_output}"

        logger.debug("Code execution finished. Saving plot to buffer...")

        # Check if a plot was actually created by the executed code
        if not plt.get_fignums():
            logger.warning("Code executed but no matplotlib figure was generated.")
            # Return the execution output and a warning, or just a warning message
            return f"Code executed successfully, but no plot was generated.\nExecution Output:\n{execution_output}"

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.close()  # Close the figure to free memory
        logger.info(
            "Data visualization created successfully (returning base64 string)."
        )
        # Optionally include execution output along with the image data URI if needed
        # return f"data:image/png;base64,{img_str}\nExecution Output:\n{execution_output}"
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        error_msg = f"Error creating chart: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Data visualization failed. Error:\n{error_msg}")
        # Return the error message instead of the base64 string
        return error_msg
