"""
Main application file for the combined FastAPI application.
This file integrates various tools including Python REPL, Data Visualization, DuckDuckGo Search,
Email, Text Anonymization, and Database Operations. # <-- Beschreibung aktualisiert
"""

import io
import base64
import logging
import os
import datetime
from typing import List, Optional, Dict, Any  # <-- Dict, Any hinzugefügt

# FastAPI imports
from fastapi import (
    FastAPI,
    Body,
    HTTPException,
    Query,
    Path,
    status,
)  # <-- Path, status hinzugefügt
from fastapi.responses import JSONResponse

# MCP imports
from fastapi_mcp import FastApiMCP

# Pydantic models
# Import Pydantic BaseModel with an alias first
# because of coflict with SQLAlchemy BaseModel
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, EmailStr


# Import tools and models from other modules
from app.mcp_search.duck_search import (
    DuckDuckGoSearcher,
    DuckDuckGoSearchResults,
)
from app.mcp_python.python_tools import PythonREPL, data_visualization
from app.mcp_email.email_sender import EmailSender
from app.mcp_anonymizer.anonymiz import Anonymizer

# --- Database Imports ---
from app.mcp_database.database import (
    DatabaseHandler,
    Base,
)  # Import Base if needed for query filters

try:
    # Versuche, das User-Modell zu importieren
    from app.mcp_database.database import User as UserModel
except ImportError:
    # Fallback
    logging.warning("Could not import User model from database.py, defining fallback.")
    from sqlalchemy import Column, Integer, String, DateTime

    # Import the SQLAlchemy BaseModel with a DISTINCT alias
    from app.mcp_database.database import BaseModel as DatabaseBaseModel

    # Define the fallback UserModel using the DISTINCT alias
    class UserModel(DatabaseBaseModel):  # <--- Use the alias here
        __tablename__ = "users"
        name = Column(String)
        email = Column(String, unique=True)

        def __repr__(self):
            return f"<User(id={self.id}, name='{self.name}', email='{self.email}')>"
# --- End Database Imports ---


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("--- Starting execution of combined application ---")
# --- End Logging Configuration ---

SENDER_EMAIL = os.getenv("EMAIL_SENDER")
if not SENDER_EMAIL:
    logger.warning(
        "EMAIL_SENDER environment variable not set. Emails might fail to send."
    )

# --- Tool Instantiation ---
repl = PythonREPL()
ddg_search = DuckDuckGoSearcher()
email_sender = EmailSender()

# Anonymizer Configuration & Instantiation
anonymizer_instance = None
try:
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
    )
    logger.info("Anonymizer initialized successfully.")
except ValueError as e:
    logger.error(
        f"Configuration error initializing Anonymizer: {e}. Anonymization endpoint might not work as expected."
    )
except OSError as e:
    logger.error(
        f"Failed to initialize Anonymizer due to spaCy model issue: {e}. Please ensure 'de_core_news_lg' is downloaded."
    )
    logger.error("Run: python -m spacy download de_core_news_lg")
except Exception as e:
    logger.error(
        f"Unexpected error initializing Anonymizer: {e}. Anonymization endpoint will be unavailable."
    )
    anonymizer_instance = None

# --- Database Handler Instantiation ---
try:
    # Verwende einen sinnvollen Datenbanknamen, z.B. den aus dem Beispiel
    db_handler = DatabaseHandler("main_database.db")
    logger.info(f"DatabaseHandler initialized successfully for main_database.db")
except Exception as e:
    logger.error(f"Failed to initialize DatabaseHandler: {e}")
    db_handler = None  # Setze auf None, damit Routen fehlschlagen können
# --- End Database Handler Instantiation ---

# --- End Tool Instantiation ---

# --- Output Directory Configuration ---
GENERATED_CODE_DIR = "generated_code"
os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
logger.info(f"Generated code will be saved to: {GENERATED_CODE_DIR}")
# --- End Output Directory ---

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Combined Tools API",
    description="Provides Python REPL, Data Visualization, DuckDuckGo Search, Email, Text Anonymization, and Database capabilities via FastAPI and MCP.",  # <-- Beschreibung aktualisiert
    version="1.0.0",
)
# --- End FastAPI App Initialization ---


# --- Pydantic Models ---
class CodeInput(PydanticBaseModel):
    code: str = Field(..., description="Python code to execute.")


class ReplOutput(PydanticBaseModel):
    output: str = Field(..., description="The standard output of the executed code.")


class VisualizationOutput(PydanticBaseModel):
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


class SearchOutput(PydanticBaseModel):
    results_markdown: str = Field(
        ..., description="Search results formatted as markdown."
    )


class EmailInput(PydanticBaseModel):
    to_address: EmailStr = Field(..., description="The recipient's email address.")
    subject: str = Field(..., description="The subject line of the email.")
    body: str = Field(..., description="The main content/body of the email.")


class EmailOutput(PydanticBaseModel):
    message: str = Field(
        ..., description="A message indicating the outcome (success or failure)."
    )


class AnonymizeInput(PydanticBaseModel):
    text: str = Field(..., description="The text to be anonymized.")


class AnonymizeOutput(PydanticBaseModel):
    anonymized_text: str = Field(
        ..., description="The anonymized version of the input text."
    )
    message: Optional[str] = Field(
        None,
        description="Additional status messages, e.g., if anonymizer wasn't initialized.",
    )


# --- Database Pydantic Models ---
class UserBase(PydanticBaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None


class UserCreate(UserBase):
    name: str  # Name ist beim Erstellen erforderlich
    email: EmailStr  # Email ist beim Erstellen erforderlich


class UserUpdate(UserBase):
    # Alle Felder sind optional beim Update
    pass


class UserResponse(UserBase):
    id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        orm_mode = True  # Erlaubt das direkte Erstellen aus SQLAlchemy-Objekten


# --- End Database Pydantic Models ---


class ErrorResponse(PydanticBaseModel):
    error: str = Field(..., description="Details about the error that occurred.")
    execution_output: Optional[str] = Field(
        None, description="Standard output captured before the error, if available."
    )


# --- End Pydantic Models ---


# --- Helper Function for Database Availability ---
def check_db_handler():
    if db_handler is None:
        logger.error("Database handler is not available.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is not initialized or available.",
        )


# --- End Helper Function ---


# --- API Endpoints ---

# ... (vorhandene Endpunkte für REPL, Visualization, Search, Email, Anonymize bleiben hier) ...


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
    tags=["Search Tools", "Weather Tools"],  # Behalte beide Tags bei, falls relevant
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

    if not SENDER_EMAIL:
        logger.error(
            "Cannot send email: EMAIL_SENDER environment variable is not configured."
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Email sending is not configured on the server (missing sender address)."
            },
        )

    try:
        success = email_sender.send_email(
            sender_email=SENDER_EMAIL,
            recipient_emails=[payload.to_address],
            subject=payload.subject,
            body=payload.body,
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


@app.post(
    "/anonymize",
    operation_id="anonymize_text_tool",
    tags=["Anonymization Tools"],
    response_model=AnonymizeOutput,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
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

    try:
        results = anonymizer_instance.anonymize_batch([payload.text])
        if results is None:
            logger.error(
                "Anonymization process failed internally within the Anonymizer tool."
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Anonymization failed during processing. Check server logs."
                },
            )
        anonymized_text = results[0]
        logger.info("Text anonymization successful.")
        return AnonymizeOutput(anonymized_text=anonymized_text)

    except Exception as e:
        error_msg = f"An unexpected error occurred during text anonymization: {e}"
        logger.exception("Exception caught in /anonymize endpoint")
        return JSONResponse(status_code=500, content={"error": error_msg})


# --- Datanbase-Routes (Example for 'User' Model) ---


@app.post(
    "/database/users",
    operation_id="create_user_in_db",
    tags=["Database Tools"],
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,  # Status Code für erfolgreiches Erstellen
    responses={
        400: {
            "model": ErrorResponse
        },  # Falls z.B. E-Mail schon existiert (wird von DB-Layer behandelt)
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},  # Falls DB nicht verfügbar
    },
)
async def create_user(user_data: UserCreate = Body(...)):
    """
    Creates a new user record in the database.
    """
    check_db_handler()  # Prüft, ob db_handler verfügbar ist
    logger.info(f"Tool 'create_user_db' called via FastAPI route.")
    try:
        # Konvertiere Pydantic-Modell zu Dict für den Handler
        user_dict = user_data.dict()
        new_user = db_handler.create(UserModel, user_dict)
        if new_user:
            logger.info(f"User created successfully with ID: {new_user.id}")
            # Konvertiere SQLAlchemy-Objekt zu Pydantic-Modell für die Antwort
            return UserResponse.from_orm(new_user)
        else:
            # Dieser Fall sollte durch Exceptions im Handler abgedeckt sein, aber zur Sicherheit:
            logger.error(
                "User creation failed for unknown reasons (db_handler.create returned None)."
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user.",
            )
    except Exception as e:
        # Fange spezifische DB-Fehler ab, falls nötig (z.B. IntegrityError für unique constraints)
        logger.exception(f"Error creating user: {e}")
        # Hier könnte man spezifischere Fehlercodes zurückgeben, z.B. 409 Conflict bei unique constraint
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}",
        )


@app.get(
    "/database/users/{user_id}",
    operation_id="read_user_from_db",
    tags=["Database Tools"],
    response_model=UserResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def read_user(
    user_id: int = Path(..., description="The ID of the user to retrieve."),
):
    """
    Retrieves a specific user record by its ID.
    """
    check_db_handler()
    logger.info(f"Tool 'read_user_db' called via FastAPI route for user ID: {user_id}")
    try:
        user = db_handler.read(UserModel, user_id)
        if user:
            logger.info(f"User found with ID: {user_id}")
            return UserResponse.from_orm(user)
        else:
            logger.warning(f"User with ID {user_id} not found.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found.",
            )
    except Exception as e:
        logger.exception(f"Error reading user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read user: {str(e)}",
        )


@app.put(
    "/database/users/{user_id}",
    operation_id="update_user_in_db",
    tags=["Database Tools"],
    response_model=UserResponse,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},  # Für ungültige Update-Daten
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def update_user(
    user_id: int = Path(..., description="The ID of the user to update."),
    user_data: UserUpdate = Body(...),
):
    """
    Updates an existing user record by its ID.
    Only provided fields will be updated.
    """
    check_db_handler()
    logger.info(
        f"Tool 'update_user_db' called via FastAPI route for user ID: {user_id}"
    )
    # Entferne Felder, die nicht gesetzt sind (None), damit der Handler sie nicht überschreibt
    update_data = user_data.dict(exclude_unset=True)

    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No update data provided."
        )

    try:
        updated_user = db_handler.update(UserModel, user_id, update_data)
        if updated_user:
            logger.info(f"User updated successfully with ID: {user_id}")
            return UserResponse.from_orm(updated_user)
        else:
            # db_handler.update gibt None zurück, wenn der User nicht existiert
            logger.warning(f"User with ID {user_id} not found for update.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found.",
            )
    except Exception as e:
        logger.exception(f"Error updating user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}",
        )


@app.delete(
    "/database/users/{user_id}",
    operation_id="delete_user_from_db",
    tags=["Database Tools"],
    status_code=status.HTTP_204_NO_CONTENT,  # Kein Body bei erfolgreichem Löschen
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def delete_user(
    user_id: int = Path(..., description="The ID of the user to delete."),
):
    """
    Deletes a user record by its ID.
    """
    check_db_handler()
    logger.info(
        f"Tool 'delete_user_db' called via FastAPI route for user ID: {user_id}"
    )
    try:
        deleted = db_handler.delete(UserModel, user_id)
        if deleted:
            logger.info(f"User deleted successfully with ID: {user_id}")
            # Bei 204 No Content wird keine Antwort gesendet
            return None  # FastAPI behandelt das korrekt
        else:
            logger.warning(f"User with ID {user_id} not found for deletion.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found.",
            )
    except Exception as e:
        logger.exception(f"Error deleting user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}",
        )


@app.get(
    "/database/users",
    operation_id="query_users_db",
    tags=["Database Tools"],
    response_model=List[UserResponse],  # Gibt eine Liste von Usern zurück
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def query_users(
    name_like: Optional[str] = Query(
        None,
        description="Filter users whose name contains this string (case-insensitive).",
    ),
    email: Optional[EmailStr] = Query(
        None, description="Filter users by exact email address."
    ),
    limit: Optional[int] = Query(
        None, description="Maximum number of users to return."
    ),
    offset: Optional[int] = Query(None, description="Number of users to skip."),
    # Man könnte hier auch order_by hinzufügen, z.B. order_by: Optional[str] = Query(None, description="Field to order by (e.g., 'name' or '-id' for descending)")
):
    """
    Queries user records with optional filtering, ordering, limit and offset.
    """
    check_db_handler()
    logger.info(
        f"Tool 'query_users_db' called via FastAPI route with filters: name_like={name_like}, email={email}, limit={limit}, offset={offset}"
    )

    filters = []
    if name_like:
        # Verwende ilike für case-insensitive Suche
        filters.append(UserModel.name.ilike(f"%{name_like}%"))
    if email:
        filters.append(UserModel.email == email)

    # Hier könnte man die order_by Logik hinzufügen, falls der Parameter existiert

    try:
        users = db_handler.query(
            UserModel,
            filters=filters if filters else None,
            # order_by=..., # Hier order_by einfügen
            limit=limit,
            offset=offset,
        )
        logger.info(f"Query returned {len(users)} users.")
        # Konvertiere jedes SQLAlchemy-Objekt in der Liste zu Pydantic
        return [UserResponse.from_orm(user) for user in users]
    except Exception as e:
        logger.exception(f"Error querying users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query users: {str(e)}",
        )


# --- END Database-Routes ---


# --- End API Endpoints ---


# --- MCP Integration ---
mcp = FastApiMCP(
    app,
    name="Combined Tools MCP",
    description="An API providing Python REPL, Data Visualization, DuckDuckGo Search, Email, Text Anonymization, and Database capabilities.",  # <-- Beschreibung aktualisiert
    base_url=f"http://{os.environ['SERVER_HOST']}:{int(os.environ['SERVER_PORT'])}",
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
