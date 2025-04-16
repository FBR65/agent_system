import os
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileSystemCrawler:
    def __init__(self, knowledge_system):
        """
        Initialisiert den FileSystemCrawler mit einer Instanz des IntegratedKnowledgeSystem.

        Args:
            knowledge_system (IntegratedKnowledgeSystem): Eine Instanz des Systems zur Wissensaufnahme.
        """
        self.knowledge_system = knowledge_system

    def crawl(self, root_dir: str, allowed_extensions: Optional[List[str]] = None):
        """
        Durchsucht rekursiv das angegebene Verzeichnis nach Dateien.

        Args:
            root_dir (str): Das Wurzelverzeichnis für die Suche.
            allowed_extensions (List[str], optional): Eine Liste von Dateiendungen (z.B. ['.pdf', '.txt']).
                                                     Wenn None, werden alle Dateitypen berücksichtigt.
        """
        logger.info(f"Starte Crawling im Verzeichnis: {root_dir}")
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isfile(item_path):
                if allowed_extensions is None or any(
                    item_path.endswith(ext) for ext in allowed_extensions
                ):
                    logger.info(f"Gefundene Datei: {item_path}")
                    try:
                        self.knowledge_system.index_document(item_path)
                    except Exception as e:
                        logger.error(
                            f"Fehler beim Verarbeiten der Datei '{item_path}': {e}"
                        )
            elif os.path.isdir(item_path):
                # Rekursiver Aufruf für Unterverzeichnisse
                self.crawl(item_path, allowed_extensions)
        logger.info(f"Crawling im Verzeichnis '{root_dir}' abgeschlossen.")


if __name__ == "__main__":
    from integrated_knowledge_system import (
        IntegratedKnowledgeSystem,
    )  # Stellen Sie sicher, dass der Importpfad korrekt ist

    # Erstellen Sie eine Instanz des IntegratedKnowledgeSystem
    collection_name = "my_crawled_knowledge"
    knowledge_system_instance = IntegratedKnowledgeSystem(
        collection_name=collection_name
    )

    # Erstellen Sie eine Instanz des FileSystemCrawlers
    crawler = FileSystemCrawler(knowledge_system=knowledge_system_instance)

    # Geben Sie das Wurzelverzeichnis an, das durchsucht werden soll
    root_directory_to_crawl = "./data_crawl"  # Erstellen Sie diesen Ordner und legen Sie einige Testdateien hinein

    # Geben Sie optional eine Liste von erlaubten Dateiendungen an
    allowed_file_types = [".pdf", ".txt"]

    # Stellen Sie sicher, dass das Wurzelverzeichnis existiert
    if not os.path.exists(root_directory_to_crawl):
        os.makedirs(root_directory_to_crawl)
        # Erstellen Sie einige leere Testdateien
        with open(os.path.join(root_directory_to_crawl, "test1.txt"), "w") as f:
            f.write("Dies ist eine Testdatei.")
        with open(os.path.join(root_directory_to_crawl, "test2.pdf"), "w") as f:
            f.write(
                "%PDF-1.4\n%\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Count 0>>endobj\nxref\n1 3\n0000000000 65535 f \n0000000009 00000 n \n0000000053 00000 n \ntrailer<</Size 3/Root 1 0 R>>startxref\n103\n%%EOF\n"
            )  # Minimal gültige PDF-Struktur
        os.makedirs(os.path.join(root_directory_to_crawl, "subfolder"), exist_ok=True)
        with open(
            os.path.join(root_directory_to_crawl, "subfolder", "test3.txt"), "w"
        ) as f:
            f.write("Dies ist eine Datei im Unterordner.")

    # Starten Sie den Crawler
    crawler.crawl(
        root_dir=root_directory_to_crawl, allowed_extensions=allowed_file_types
    )

    # Nach dem Crawling können Sie das Wissen abfragen
    query = "Was ist der Inhalt von test1?"
    results = knowledge_system_instance.query_knowledge(query, limit=5)
    print("\nSuchergebnisse nach dem Crawling:")
    for result in results:
        print(f"  {result.get('text', 'Kein Text')}")
