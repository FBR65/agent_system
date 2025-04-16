from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from tika import parser
from datetime import datetime as dt
from sklearn.metrics.pairwise import cosine_similarity
import logging
import secrets
import uuid
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
from openai import OpenAI
from typing import List, Dict, Optional

# Konstanten (könnten auch in der __init__ konfigurierbar gemacht werden)
TIKA_SERVER_URL = "http://127.0.0.1:9998/tika"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 120
EMBEDDING_MODEL = "paraphrase-multilingual:latest"
SPARSE_EMBEDDING_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
SEMANTIC_CHUNK_MODEL = "aari1995/German_Semantic_V3"
SPACY_MODEL = "de_core_news_sm"
DISTANCE_METRIC = "MANHATTAN"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globale Variablen für spaCy und SentenceTransformer (einmal laden)
nlp = spacy.load(SPACY_MODEL)
model = SentenceTransformer(SEMANTIC_CHUNK_MODEL, trust_remote_code=True)


class Document:
    """
    Strukturierte Ausgabe für das geparste Dokument.
    """

    def __init__(self, inhalt: str, metadaten: dict = None):
        self.inhalt = inhalt
        self.metadaten = metadaten if metadaten is not None else {}


def semantic_chunking(text, threshold_percentile=25):
    # Schritt 1: Sätze aufteilen
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Schritt 2: Embeddings generieren
    embeddings = model.encode(sentences)

    # Schritt 3: Ähnlichkeiten berechnen
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        similarities.append(sim)

    # Schritt 4: Breakpoints ermitteln
    threshold = np.percentile(similarities, threshold_percentile)

    # Schritt 5: Chunks erstellen
    chunks, current_chunk = [], []
    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        if i < len(similarities) and similarities[i] < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class IntegratedKnowledgeSystem:
    def __init__(self, collection_name):
        """
        Initialisiert das IntegratedKnowledgeSystem mit dem Namen der Qdrant-Collection.
        """
        self.collection_name = f"{collection_name}_{DISTANCE_METRIC.upper()}"
        self.sparse_embedding_model = SparseTextEmbedding(
            model_name=SPARSE_EMBEDDING_MODEL_NAME
        )
        self.openai_client = OpenAI(
            base_url="http://localhost:11434/v1/", api_key="my key"
        )
        self.qdrant_client = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT
        )
        self._create_collection()

    def _create_collection(self):
        """
        Erstellt die Qdrant-Collection, falls sie noch nicht existiert, mit der MANHATTAN-Distanz.
        """
        if not self.qdrant_client.collection_exists(
            collection_name=self.collection_name
        ):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=768, distance=models.Distance.MANHATTAN
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True,
                ),
            )
            logging.info(f"Collection '{self.collection_name}' created.")
        else:
            logging.info(f"Collection '{self.collection_name}' already exists.")

    def generate_point_id(self):
        """
        Generiert eine eindeutige Point-ID.
        """
        uuid_value = uuid.uuid4().hex
        modified_uuid = "".join(
            (
                hex((int(c, 16) ^ secrets.randbits(4) & 15 >> int(c) // 4))[2:]
                if c in "018"
                else c
            )
            for c in uuid_value
        )
        logging.info(f"Created point id '{modified_uuid}'.")
        return str(modified_uuid)

    def stream_document(self, file_path):
        """
        Parsen des Dokuments mit Tika.
        """
        logging.info(f"Streaming document: {file_path}")
        parsed = parser.from_file(file_path, serverEndpoint=TIKA_SERVER_URL)
        metadata = parsed.get("metadata", {})
        if "resourceName" in metadata:
            resource_name = metadata["resourceName"]
            if isinstance(resource_name, list):
                metadata["file_name"] = resource_name[0].strip("b'")
            else:
                metadata["file_name"] = resource_name.strip("b'")
            del metadata["resourceName"]
        content = parsed.get("content", "")
        return Document(inhalt=content, metadaten=metadata)

    def split_into_chunks(self, text, output_dir="temp_chunks"):
        """
        Teilt den Text in semantische Chunks auf.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        chunk_files = []
        chunks = semantic_chunking(text)
        for i, chunk in enumerate(chunks):
            chunk_filename = os.path.join(output_dir, f"chunk_{i}.json")
            with open(chunk_filename, "w", encoding="utf-8") as f:
                json.dump({"text": chunk}, f)
            chunk_files.append(chunk_filename)
        logging.info(f"Saved {len(chunk_files)} chunks to '{output_dir}'.")
        return chunk_files

    def index_document(self, file_path):
        """
        Verarbeitet ein Dokument, teilt es in Chunks auf und speichert diese in der Qdrant-Datenbank.
        """
        logging.info(f"Indexing document: {file_path}")
        document = self.stream_document(file_path)
        chunk_files = self.split_into_chunks(document.inhalt)
        self._fill_database(chunk_files)
        # Bereinigung der temporären Chunk-Dateien
        for file_path in chunk_files:
            os.remove(file_path)
        if os.path.exists("temp_chunks") and not os.listdir("temp_chunks"):
            os.rmdir("temp_chunks")
        logging.info(f"Document '{os.path.basename(file_path)}' indexed successfully.")

    def _fill_database(self, chunk_file_paths):
        """
        Liest Chunk-Dateien, erstellt Embeddings und speichert sie in Qdrant.
        """
        points = []
        for file_path in chunk_file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunk_data = json.load(f)
                    chunk = chunk_data["text"]

                response = self.openai_client.embeddings.create(
                    input=[chunk], model=EMBEDDING_MODEL
                )
                dense_embedding = response.data[0].embedding

                sparse_embedding_result = self.sparse_embedding_model.embed([chunk])
                sparse_embedding = None
                if sparse_embedding_result:
                    embedding = sparse_embedding_result[0]
                    sparse_embedding = {
                        "indices": embedding.indices.tolist(),
                        "values": embedding.values.tolist(),
                    }

                point_id = self.generate_point_id()
                payload = {"text": chunk, "chunk_id": point_id}
                if dense_embedding and sparse_embedding:
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector={
                                "dense": dense_embedding,
                                "sparse": sparse_embedding,
                            },
                            payload=payload,
                        )
                    )
                elif dense_embedding:
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector={"dense": dense_embedding},
                            payload=payload,
                        )
                    )
                elif sparse_embedding:
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector={"sparse": sparse_embedding},
                            payload=payload,
                        )
                    )
                else:
                    logging.warning(
                        f"Could not create embeddings for chunk: '{chunk[:50]}...'"
                    )

            except Exception as e:
                logging.error(f"Error processing chunk file '{file_path}': {e}")

        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )
            logging.info(
                f"Successfully uploaded {len(points)} chunks to '{self.collection_name}'."
            )
        else:
            logging.info("No data to upload.")

    def query_knowledge(
        self, query: str, limit: int = 10, metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Führt eine Suche in der Qdrant-Datenbank durch und gibt die Payloads der Ergebnisse zurück.
        Verwendet sowohl dichte als auch spärliche Vektoren für die Suche (mit RRF Fusion).

        Args:
            query (str): Die Suchanfrage.
            limit (int, optional): Die maximale Anzahl der zurückzugebenden Ergebnisse. Defaults to 10.
            metadata_filter (Dict, optional): Ein optionaler Filter basierend auf Metadaten. Defaults to None.

        Returns:
            List[Dict]: Eine Liste der Payloads der relevantesten Knowledge-Einheiten.
        """
        response = self.openai_client.embeddings.create(
            input=[query], model=EMBEDDING_MODEL
        )
        dense_query = response.data[0].embedding
        sparse_query_result = self.sparse_embedding_model.embed([query])
        sparse_query = sparse_query_result[0] if sparse_query_result else None

        search_params = models.SearchParams(
            hnsw_ef=128
        )  # Optional: Anpassung der Suchparameter

        try:
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=dense_query,
                query_sparse_vector=models.SparseVector(
                    indices=sparse_query.indices.tolist(),
                    values=sparse_query.values.tolist(),
                )
                if sparse_query
                else None,
                limit=limit,
                query_filter=models.Filter(
                    **{
                        "must": [
                            {"key": key, "match": {"value": value}}
                            for key, value in metadata_filter.items()
                        ]
                    }
                )
                if metadata_filter
                else None,
                with_payload=True,
                search_params=search_params,
                score_threshold=0.0,  # Optional: Mindest-Score für Ergebnisse
            )
            return [hit.payload for hit in search_result]
        except Exception as e:
            logging.error(f"Error during query: {e}")
            return []

    def get_specific_knowledge(self, point_ids: List[str]) -> List[Dict]:
        """
        Ruft spezifische Knowledge-Einheiten anhand ihrer IDs ab.

        Args:
            point_ids (List[str]): Eine Liste der IDs der abzurufenden Knowledge-Einheiten.

        Returns:
            List[Dict]: Eine Liste der Payloads der abgerufenen Knowledge-Einheiten.
        """
        points = self.qdrant_client.get_points(
            collection_name=self.collection_name, ids=point_ids, with_payload=True
        )
        return [point.payload for point in points]


if __name__ == "__main__":
    collection_name = "integrated_knowledge_test"
    knowledge_system = IntegratedKnowledgeSystem(collection_name=collection_name)

    # Pfad zu Ihrer Testdatei
    file_path = "./data/datenstrategie.pdf"
    if os.path.exists(file_path):
        knowledge_system.index_document(file_path)
    else:
        print(f"Datei nicht gefunden: {file_path}")

    # Beispielhafte Suchanfrage
    query = "Welche Vorteile hat die Datenstrategie?"
    results = knowledge_system.query_knowledge(query, limit=5)
    print("\nSuchergebnisse:")
    for result in results:
        print(f"  {result.get('text', 'Kein Text')}")

    # Beispielhaftes Abrufen spezifischer IDs (ggf. aus vorherigen Suchergebnissen)
    # specific_ids = ["...", "..."]
    # specific_knowledge = knowledge_system.get_specific_knowledge(specific_ids)
    # print("\nSpezifisches Wissen:")
    # for item in specific_knowledge:
    #     print(f"  {item.get('text', 'Kein Text')}")
