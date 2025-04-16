# c:\Users\frank\Documents\py_projects\dev\agent_system\app\app.py
import io
import base64
import logging
import os
import datetime
from typing import List, Optional

# FastAPI imports
from fastapi import FastAPI, Body, HTTPException, Query
from fastapi.responses import JSONResponse

# MCP imports
from fastapi_mcp import FastApiMCP

# Pydantic models
from pydantic import BaseModel, Field, EmailStr

# Import tools and models from other modules
from app.mcp_search.duck_search import (
    DuckDuckGoSearcher,
    DuckDuckGoSearchResults,
)
from app.mcp_python.python_tools import PythonREPL, data_visualization
from app.mcp_email.email_sender import EmailSender
from app.mcp_anonymizer.anonymiz import Anonymizer

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("--- Starting execution of combined application ---")
# --- End Logging Configuration ---

# --- Tool Instantiation ---
repl = PythonREPL()
ddg_search = DuckDuckGoSearcher()
email_sender = EmailSender()

# Anonymizer Configuration & Instantiation
anonymizer_instance = None
try:
    # Konfiguration über Umgebungsvariablen (optional, Anonymizer hat Defaults/Logik)
    # Beispiel: USE_LLM_ANONYMIZER = os.getenv("USE_LLM_ANONYMIZER", "false").lower() == "true"
    # Der Anonymizer selbst liest detailliertere LLM-Vars, falls use_llm=True ist.
    # Hier entscheiden wir nur, OB LLM genutzt werden soll (kann auch hardcoded sein).
    USE_LLM_FOR_ANONYMIZER = (
        False  # Setze auf True, wenn LLM genutzt werden soll und konfiguriert ist
    )
    LLM_API_KEY = os.getenv(
        "API_KEY"
    )  # Anonymizer braucht diese, wenn USE_LLM_FOR_ANONYMIZER=True
    LLM_ENDPOINT = os.getenv("BASE_URL")
    LLM_MODEL = os.getenv("MODEL_NAME")

    logger.info(
        f"Attempting to initialize Anonymizer (LLM usage: {USE_LLM_FOR_ANONYMIZER})..."
    )
    anonymizer_instance = Anonymizer(
        use_llm=USE_LLM_FOR_ANONYMIZER,
        llm_api_key=LLM_API_KEY,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_model_name=LLM_MODEL,
        # labels_to_anonymize kann hier auch überschrieben werden, wenn gewünscht
    )
    logger.info("Anonymizer initialized successfully.")
except ValueError as e:
    # Fängt Konfigurationsfehler vom Anonymizer ab (z.B. fehlender Key bei use_llm=True)
    logger.error(
        f"Configuration error initializing Anonymizer: {e}. Anonymization endpoint might not work as expected."
    )
except OSError as e:
    # Fängt Fehler ab, wenn das spaCy-Modell nicht gefunden wird
    logger.error(
        f"Failed to initialize Anonymizer due to spaCy model issue: {e}. Please ensure 'de_core_news_lg' is downloaded."
    )
    logger.error("Run: python -m spacy download de_core_news_lg")
except Exception as e:
    # Fängt andere unerwartete Fehler bei der Initialisierung ab
    logger.error(
        f"Unexpected error initializing Anonymizer: {e}. Anonymization endpoint will be unavailable."
    )
    anonymizer_instance = (
        None  # Stelle sicher, dass es None ist, wenn die Initialisierung fehlschlägt
    )

# --- End Tool Instantiation ---

# --- Output Directory Configuration ---
GENERATED_CODE_DIR = "generated_code"
os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
logger.info(f"Generated code will be saved to: {GENERATED_CODE_DIR}")
# --- End Output Directory ---

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Combined Tools API",
    description="Provides Python REPL, Data Visualization, DuckDuckGo Search, Email, and Text Anonymization capabilities via FastAPI and MCP.",  # <-- Beschreibung aktualisiert
    version="1.0.0",
)
# --- End FastAPI App Initialization ---


# --- Pydantic Models ---
class CodeInput(BaseModel):
    code: str = Field(..., description="Python code to execute.")


class ReplOutput(BaseModel):
    output: str = Field(..., description="The standard output of the executed code.")


class VisualizationOutput(BaseModel):
    image_data_url: Optional[str] = Field(
        None,
        description="Base64 encoded PNG image data URL (data:image/png;base64,...). Present on success.",
    )
    message: Optional[str] = Field(
        None,
        description="A message indicating outcome, e.g., if no plot was generated.",
    )
    execution_output: Optional[str] = Field(
        None, description="Standard output captured during code execution."
    )


class SearchOutput(BaseModel):
    results_markdown: str = Field(
        ..., description="Search results formatted as markdown."
    )


class EmailInput(BaseModel):
    to_address: EmailStr = Field(..., description="The recipient's email address.")
    subject: str = Field(..., description="The subject line of the email.")
    body: str = Field(..., description="The main content/body of the email.")


class EmailOutput(BaseModel):
    message: str = Field(
        ..., description="A message indicating the outcome (success or failure)."
    )


# --- NEUE Pydantic-Modelle für Anonymisierung ---
class AnonymizeInput(BaseModel):
    text: str = Field(..., description="The text to be anonymized.")
    # Optional: Flag hinzufügen, um Mapping pro Anfrage zurückzusetzen? Standard: Kein Reset.
    # reset_mapping: bool = Field(False, description="Reset the anonymization mapping before processing this text.")


class AnonymizeOutput(BaseModel):
    anonymized_text: str = Field(
        ..., description="The anonymized version of the input text."
    )
    # Optional: Mapping zurückgeben? Könnte groß werden.
    # mapping: Optional[dict] = Field(None, description="The mapping used for anonymization in this request (if applicable).")
    message: Optional[str] = Field(
        None,
        description="Additional status messages, e.g., if anonymizer wasn't initialized.",
    )


# --- Ende NEUE Pydantic-Modelle ---


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Details about the error that occurred.")
    execution_output: Optional[str] = Field(
        None, description="Standard output captured before the error, if available."
    )


# --- End Pydantic Models ---


# --- API Endpoints ---

# ... (bestehende Endpunkte /python-repl, /data-visualization, /search, /send-email bleiben unverändert) ...


@app.post(
    "/python-repl",
    operation_id="python_repl_tool",
    tags=["Python Tools"],
    response_model=ReplOutput,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def run_python_repl(payload: CodeInput = Body(...)):
    """
    Execute Python code in a restricted REPL environment.
    """
    logger.info("Tool 'python_repl_tool' called via FastAPI route.")
    try:
        result = repl.run(payload.code)
        if result.strip().startswith("Error:"):
            logger.error(f"REPL execution failed internally: {result}")
            return JSONResponse(status_code=400, content={"error": result})
        logger.info("REPL execution successful.")
        return ReplOutput(output=result)
    except Exception as e:
        error_msg = f"Internal server error during REPL execution: {str(e)}"
        logger.exception("Exception caught in /python-repl endpoint")
        return JSONResponse(status_code=500, content={"error": error_msg})


@app.post(
    "/data-visualization",
    operation_id="data_visualization_tool",
    tags=["Python Tools"],
    response_model=VisualizationOutput,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def create_data_visualization(payload: CodeInput = Body(...)):
    """
    Execute Python code to generate a data visualization using matplotlib.
    """
    code = payload.code
    logger.info("Tool 'data_visualization_tool' called via FastAPI route.")
    try:
        result = await data_visualization(code=code, repl=repl)
        if result.startswith("data:image/png;base64,"):
            logger.info("Data visualization created successfully by tool.")
            return VisualizationOutput(
                image_data_url=result, message="Plot generated successfully."
            )
        elif "no plot was generated" in result.lower():
            logger.warning("Data visualization tool reported no plot generated.")
            exec_output = None
            if "Execution Output:" in result:
                try:
                    exec_output = result.split("Execution Output:", 1)[1].strip()
                except IndexError:
                    pass
            return VisualizationOutput(message=result, execution_output=exec_output)
        else:
            logger.error(f"Data visualization tool failed: {result}")
            exec_output = None
            if "Execution Output:" in result:
                try:
                    exec_output = result.split("Execution Output:", 1)[1].strip()
                except IndexError:
                    pass
            status_code = 400 if "Code execution failed:" in result else 500
            return JSONResponse(
                status_code=status_code,
                content={"error": result, "execution_output": exec_output},
            )
    except Exception as e:
        error_msg = f"Internal server error during data visualization: {str(e)}"
        logger.exception("Exception caught in /data-visualization endpoint")
        return JSONResponse(status_code=500, content={"error": error_msg})


# --- DuckDuckGo Search Functionality ---
search_config = {"parameters": {"default_num_results": 5}}


def format_duckduckgo_results(search_results: DuckDuckGoSearchResults) -> str:
    """Formats DuckDuckGoSearchResults into a markdown string."""
    if not search_results.results:
        return "No search results found."
    markdown_output = ""
    for i, result in enumerate(search_results.results):
        markdown_output += f"### Result {i + 1}:\n"
        markdown_output += f"**Title:** {result.title}\n"
        markdown_output += f"**Link:** {result.link}\n"
        markdown_output += f"**Snippet:** {result.snippet}\n\n"
    return markdown_output.strip()


@app.get(
    "/search",
    operation_id="search_duckduckgo_tool",
    tags=["Search Tools, Weather Tools"],
    response_model=SearchOutput,
    responses={500: {"model": ErrorResponse}},
)
async def search_duckduckgo_route(
    query: str = Query(..., description="The search query string."),
    num_results: Optional[int] = Query(
        None, description="Number of results to return."
    ),
):
    """
    Search the web using DuckDuckGo and return results as markdown formatted text.
    """
    logger.info(
        f"Tool 'search_duckduckgo_tool' called via FastAPI route with query: '{query}', num_results: {num_results}"
    )
    try:
        actual_num_results = (
            num_results
            if num_results is not None and num_results > 0
            else search_config["parameters"]["default_num_results"]
        )
        logger.debug(f"Effective number of results: {actual_num_results}")
        search_results: DuckDuckGoSearchResults = ddg_search.search(
            query, num_results=actual_num_results
        )
        formatted_results = format_duckduckgo_results(search_results)
        logger.info(
            f"Search completed for query: '{query}'. Returning formatted results."
        )
        return SearchOutput(results_markdown=formatted_results)
    except Exception as e:
        error_msg = f"An error occurred in the search route: {e}"
        logger.exception(f"Error in 'search_duckduckgo_route' for query '{query}'")
        return JSONResponse(status_code=500, content={"error": error_msg})


@app.post(
    "/send-email",
    operation_id="send_email_tool",
    tags=["Email Tools"],
    response_model=EmailOutput,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def send_email_route(payload: EmailInput = Body(...)):
    """
    Sends an email using the configured EmailSender tool.
    """
    logger.info(
        f"Tool 'send_email_tool' called via FastAPI route to send email to {payload.to_address}"
    )
    try:
        success = email_sender.send_email(
            to_address=payload.to_address, subject=payload.subject, body=payload.body
        )
        if success:
            logger.info(f"Email successfully sent to {payload.to_address}")
            return EmailOutput(
                message=f"Email successfully sent to {payload.to_address}."
            )
        else:
            logger.error(
                f"EmailSender failed to send email to {payload.to_address}. Check EmailSender logs for details."
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Failed to send email. Internal server error or configuration issue."
                },
            )
    except ValueError as ve:
        logger.error(f"Validation error sending email: {ve}")
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        error_msg = f"An unexpected error occurred while trying to send email: {e}"
        logger.exception(
            f"Error in 'send_email_route' for recipient {payload.to_address}"
        )
        return JSONResponse(status_code=500, content={"error": error_msg})


# --- NEUE Anonymisierungs-Route ---
@app.post(
    "/anonymize",
    operation_id="anonymize_text_tool",  # <-- MCP operation_id
    tags=["Anonymization Tools"],  # <-- Tag für Gruppierung
    response_model=AnonymizeOutput,
    responses={
        400: {
            "model": ErrorResponse
        },  # Für ungültige Eingaben (obwohl Pydantic viel abfängt)
        500: {"model": ErrorResponse},  # Für interne Serverfehler
        503: {
            "model": ErrorResponse
        },  # Service Unavailable (wenn Anonymizer nicht initialisiert wurde)
    },
)
async def anonymize_text_route(payload: AnonymizeInput = Body(...)):
    """
    Anonymizes the provided text using spaCy and optionally an LLM.
    Requires the spaCy model 'de_core_news_lg' to be downloaded.
    LLM usage depends on server configuration.
    """
    logger.info("Tool 'anonymize_text_tool' called via FastAPI route.")

    if anonymizer_instance is None:
        logger.error("Anonymizer tool is not available due to initialization failure.")
        return JSONResponse(
            status_code=503,  # Service Unavailable
            content={
                "error": "Anonymizer service is not available. Check server logs for initialization errors (e.g., missing spaCy model)."
            },
        )

    # Optional: Mapping zurücksetzen, falls implementiert und gewünscht
    # if payload.reset_mapping:
    #     anonymizer_instance.reset()
    #     logger.info("Anonymization mapping reset for this request.")

    try:
        # anonymize_batch erwartet eine Liste, auch für einzelnen Text
        results = anonymizer_instance.anonymize_batch([payload.text])

        # Prüfe, ob die Methode erfolgreich war (gibt Liste zurück, sonst None bei Fehlern)
        if results is None:
            logger.error(
                "Anonymization process failed internally within the Anonymizer tool."
            )
            # Anonymizer sollte Details geloggt haben
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Anonymization failed during processing. Check server logs."
                },
            )

        # Extrahiere das Ergebnis für den einzelnen Text
        anonymized_text = results[0]
        logger.info("Text anonymization successful.")

        # Optional: Mapping zurückgeben, falls implementiert und gewünscht
        # current_mapping = anonymizer_instance.get_mapping() if include_mapping_flag else None
        return AnonymizeOutput(
            anonymized_text=anonymized_text
        )  # , mapping=current_mapping)

    except Exception as e:
        # Fange unerwartete Fehler während des Anonymisierungsaufrufs ab
        error_msg = f"An unexpected error occurred during text anonymization: {e}"
        logger.exception("Exception caught in /anonymize endpoint")
        return JSONResponse(status_code=500, content={"error": error_msg})


# --- Ende NEUE Anonymisierungs-Route ---


# --- End API Endpoints ---


# --- MCP Integration ---
mcp = FastApiMCP(
    app,
    name="Combined Tools MCP",
    description="An API providing Python REPL, Data Visualization, DuckDuckGo Search, Email, and Text Anonymization capabilities.",  # <-- Beschreibung aktualisiert
    base_url=f"{os.environ['SERVER_HOST']}:{int(os.environ['SERVER_PORT'])}",
)

# MCP Server Endpunkte einbinden (z.B. /mcp.json)
mcp.mount()
logger.info("MCP server mounted.")
# --- End MCP Integration ---


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server with Uvicorn.")
    # Modulpfad für Uvicorn verwenden: 'app:app', wenn die Datei app.py heißt
    uvicorn.run("app:app", host="0.0.0.0", port=8504, reload=True)
    logger.info("FastAPI server stopped.")
# --- End Main Execution ---
