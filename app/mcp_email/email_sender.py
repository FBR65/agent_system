import requests
import os
import base64
from dotenv import load_dotenv
import logging  # Using logging instead of print for better practice

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EmailSender:
    def __init__(self):
        """
        Initialisiert die EmailSender Klasse für die Verwendung mit einer E-Mail API (z.B. MailerSend).
        Lädt API-Schlüssel und URL aus Umgebungsvariablen (.env Datei).

        Raises:
            ValueError: Wenn EMAIL_KEY oder EMAIL_URL nicht in der .env Datei gefunden werden.
        """
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv("EMAIL_KEY")
        # Default to MailerSend URL if not provided, but log a warning.
        self.api_url = os.getenv("EMAIL_URL", "https://api.mailersend.com/v1/email")

        if not self.api_key:
            logging.error("EMAIL_KEY nicht in der .env Datei gefunden.")
            raise ValueError("EMAIL_KEY nicht in der .env Datei gefunden.")
        if not os.getenv("EMAIL_URL"):
            logging.warning(
                "EMAIL_URL nicht in der .env Datei gefunden. Standardwert '%s' wird verwendet.",
                self.api_url,
            )

    def _prepare_recipients(self, emails):
        """
        Formatiert eine Liste von E-Mail-Adressen in das von der API erwartete Format.
        Hinweis: Dieses einfache Format unterstützt keine individuellen Namen pro Empfänger.
        Für Namen müssten die Eingabeparameter angepasst werden (z.B. Liste von Dictionaries).
        """
        if not emails:
            return []
        # Convert list of email strings to list of dicts expected by API
        return [{"email": email} for email in emails]

    def _prepare_attachments(self, attachment_paths):
        """Bereitet Anhänge für die API vor (Base64-kodiert)."""
        attachments_payload = []
        if attachment_paths:
            for attachment_path in attachment_paths:
                if not os.path.isfile(attachment_path):
                    logging.warning(
                        f"Anhangdatei nicht gefunden oder ist kein File: {attachment_path}. Wird übersprungen."
                    )
                    continue  # Skip this file
                try:
                    with open(attachment_path, "rb") as attachment_file:
                        content = attachment_file.read()
                        encoded_content = base64.b64encode(content).decode("utf-8")
                        attachments_payload.append(
                            {
                                "filename": os.path.basename(attachment_path),
                                "content": encoded_content,
                            }
                        )
                except IOError as e:
                    logging.warning(
                        f"Fehler beim Lesen der Anhangdatei {attachment_path}: {e}. Wird übersprungen."
                    )
                except Exception as e:
                    logging.warning(
                        f"Unerwarteter Fehler beim Verarbeiten des Anhangs {attachment_path}: {e}. Wird übersprungen."
                    )
        return attachments_payload

    def send_email(
        self,
        sender_email,
        recipient_emails,
        subject,
        body,
        sender_name=None,  # Optionaler Absendername
        content_type="plain",
        attachments=None,  # Liste von Dateipfaden
        cc_emails=None,
        bcc_emails=None,
    ):
        """
        Sendet eine E-Mail über die konfigurierte E-Mail API.

        Args:
            sender_email (str): Die E-Mail-Adresse des Absenders.
            recipient_emails (list): Eine Liste der E-Mail-Adressen der Empfänger.
            subject (str): Der Betreff der E-Mail.
            body (str): Der Inhalt der E-Mail (Text oder HTML).
            sender_name (str, optional): Der Name des Absenders. Defaults to None.
            content_type (str, optional): Der Inhaltstyp ('plain' für Text oder 'html' für HTML). Defaults to 'plain'.
            attachments (list, optional): Eine Liste von Dateipfaden der Anhänge. Defaults to None.
            cc_emails (list, optional): Eine Liste der E-Mail-Adressen für CC. Defaults to None.
            bcc_emails (list, optional): Eine Liste der E-Mail-Adressen für BCC. Defaults to None.

        Returns:
            bool: True, wenn die E-Mail erfolgreich über die API gesendet wurde, False bei einem Fehler.
        """
        if not self.api_key or not self.api_url:
            logging.error("API-Schlüssel oder URL nicht konfiguriert.")
            return False

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        from_data = {"email": sender_email}
        if sender_name:
            from_data["name"] = sender_name

        # Grundstruktur des Payloads
        payload = {
            "from": from_data,
            "to": self._prepare_recipients(recipient_emails),
            "subject": subject,
        }

        # Füge CC und BCC hinzu, falls vorhanden
        if cc_emails:
            payload["cc"] = self._prepare_recipients(cc_emails)
        if bcc_emails:
            payload["bcc"] = self._prepare_recipients(bcc_emails)

        # Füge Text- oder HTML-Inhalt hinzu
        if content_type == "plain":
            payload["text"] = body
        elif content_type == "html":
            payload["html"] = body
        else:
            logging.error(
                "Ungültiger Content-Type: '%s'. Muss 'plain' oder 'html' sein.",
                content_type,
            )
            return False  # Fehler bei ungültigem Typ

        # Bereite Anhänge vor und füge sie hinzu
        prepared_attachments = self._prepare_attachments(attachments)
        if prepared_attachments:
            payload["attachments"] = prepared_attachments

        # Sende die Anfrage an die API
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)

            # Überprüfe auf HTTP-Fehler (4xx oder 5xx)
            response.raise_for_status()

            # MailerSend gibt bei Erfolg 202 Accepted zurück
            if response.status_code == 202:
                message_id = response.headers.get("X-Message-Id", "N/A")
                logging.info(
                    f"E-Mail erfolgreich an API übermittelt. Status: {response.status_code}. Message-ID: {message_id}"
                )
                return True
            else:
                # Sollte durch raise_for_status() abgedeckt sein, aber als Fallback
                logging.warning(
                    f"Unerwarteter Erfolgsstatus von API: {response.status_code} - {response.text}"
                )
                return False  # Nicht der erwartete Erfolgscode

        except requests.exceptions.HTTPError as e:
            # Spezifische Behandlung von HTTP-Fehlern (z.B. 401, 422, 500)
            error_details = "Keine Details verfügbar."
            try:
                # Versuche, Fehlerdetails aus der JSON-Antwort zu extrahieren
                error_data = e.response.json()
                error_details = error_data.get("message", str(error_data))
            except ValueError:  # JSONDecodeError
                error_details = e.response.text  # Falls Antwort kein JSON ist
            logging.error(
                f"HTTP-Fehler bei API-Anfrage an {e.request.url}: {e.response.status_code} {e.response.reason}. Details: {error_details}"
            )
            return False
        except requests.exceptions.RequestException as e:
            # Andere Request-Fehler (z.B. Timeout, Verbindungsproblem)
            logging.error(f"Fehler bei der API-Anfrage: {e}")
            return False
        except Exception as e:
            # Andere unerwartete Fehler
            logging.exception(f"Unerwarteter Fehler beim Senden der E-Mail: {e}")
            return False


# Beispielhafte Verwendung (aktualisiert)
if __name__ == "__main__":
    logging.info("Starte Beispiel-E-Mail-Versand über API...")

    # Temporäre Testdateien erstellen
    attachment_files = ["api_test.txt", "api_test.pdf"]
    try:
        with open(attachment_files[0], "w", encoding="utf-8") as f:
            f.write("Dies ist eine Test-Datei für die API.")
        # Minimal valid PDF structure
        with open(attachment_files[1], "w", encoding="utf-8") as f:  # Specify encoding
            f.write(
                "%PDF-1.4\n%\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Count 0>>endobj\nxref\n1 3\n0000000000 65535 f \n0000000009 00000 n \n0000000053 00000 n \ntrailer<</Size 3/Root 1 0 R>>startxref\n103\n%%EOF\n"
            )
        logging.info("Testdateien erstellt: %s", attachment_files)
    except IOError as e:
        logging.error(f"Fehler beim Erstellen der Testdateien: {e}")
        attachment_files = []  # Keine Anhänge verwenden, wenn Erstellung fehlschlägt

    try:
        # EmailSender wird mit Werten aus .env initialisiert
        email_sender = EmailSender()

        # Konfiguration - BITTE ANPASSEN!
        # Stellen Sie sicher, dass sender_email eine in Ihrem E-Mail-Provider (z.B. MailerSend) verifizierte Domain/Adresse ist.
        sender = os.getenv("EMAIL_SENDER")
        sender_name = "API Test Sender"
        recipients = ["fbr65@duck.com"]  # Testempfänger
        # cc = ["fbr65@duck.com"]
        # bcc = ["fbr65@duck.com"]  # BCC wird vom API-Provider verarbeitet
        subject = "API Test-E-Mail mit Anhängen und CC/BCC"
        body_plain = "Dies ist der Inhalt der E-Mail im Textformat.\n\nMit freundlichen Grüßen,\nIhr API Agent"
        body_html = """<html><body><p>Dies ist der Inhalt der E-Mail im <b>HTML</b>-Format.</p><br><p>Mit freundlichen Grüßen,<br>Ihr API Agent</p></body></html>"""

        logging.info("Sende E-Mail (Text) über API...")
        success_plain = email_sender.send_email(
            sender_email=sender,
            sender_name=sender_name,
            recipient_emails=recipients,
            subject=subject + " (Plain)",
            body=body_plain,
            content_type="plain",
            attachments=attachment_files,
            # cc_emails=cc,
            # bcc_emails=bcc,
        )
        if success_plain:
            logging.info("E-Mail (Text) erfolgreich über API gesendet!")
        else:
            logging.error("Fehler beim Senden der E-Mail (Text) über API.")

        logging.info("\nSende E-Mail (HTML) über API...")
        success_html = email_sender.send_email(
            sender_email=sender,
            sender_name=sender_name,
            recipient_emails=recipients,
            subject=subject + " (HTML)",
            body=body_html,
            content_type="html",
            attachments=attachment_files,
            # cc_emails=cc,
            # bcc_emails=bcc,
        )
        if success_html:
            logging.info("E-Mail (HTML) erfolgreich über API gesendet!")
        else:
            logging.error("Fehler beim Senden der E-Mail (HTML) über API.")

    except ValueError as e:
        # Fängt Fehler von __init__ ab (fehlende .env Variablen)
        logging.error(f"Konfigurationsfehler: {e}")
    except Exception as e:
        logging.exception(f"Ein unerwarteter Fehler ist im Beispiel aufgetreten: {e}")
    finally:
        # Aufräumen der Test-Dateien
        logging.info("Räume Testdateien auf...")
        for f_path in attachment_files:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    logging.info(f"Testdatei {f_path} entfernt.")
                except OSError as e:
                    logging.error(f"Fehler beim Entfernen der Testdatei {f_path}: {e}")
