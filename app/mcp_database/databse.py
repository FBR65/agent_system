from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class BaseModel(Base):
    __abstract__ = True
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatabaseHandler:
    def __init__(self, db_connection_string):
        """
        Initialisiert den DatabaseHandler mit der Datenbank-Connection-String.

        Args:
            db_connection_string (str): Die SQLAlchemy-kompatible Connection-String
                                         (z.B. 'sqlite:///./mydatabase.db', 'postgresql://user:password@host:port/database').
        """
        self.engine = create_engine(db_connection_string)
        Base.metadata.create_all(self.engine)  # Erstellt die Tabellen in der Datenbank
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create(self, model, data):
        """
        Erstellt einen neuen Eintrag in der Datenbank.

        Args:
            model (SQLAlchemy Model): Die SQLAlchemy-Modellklasse für die Tabelle.
            data (dict): Ein Dictionary mit den Daten für den neuen Eintrag.

        Returns:
            The newly created model instance or None if an error occurred.
        """
        session = self.SessionLocal()
        try:
            instance = model(**data)
            session.add(instance)
            session.commit()
            session.refresh(instance)
            return instance
        except Exception as e:
            session.rollback()
            print(f"Fehler beim Erstellen des Eintrags: {e}")
            return None
        finally:
            session.close()

    def read(self, model, item_id):
        """
        Liest einen Eintrag anhand seiner ID.

        Args:
            model (SQLAlchemy Model): Die SQLAlchemy-Modellklasse für die Tabelle.
            item_id (int): Die ID des zu lesenden Eintrags.

        Returns:
            The model instance with the given ID or None if not found.
        """
        session = self.SessionLocal()
        try:
            return session.query(model).filter(model.id == item_id).first()
        finally:
            session.close()

    def update(self, model, item_id, data):
        """
        Aktualisiert einen bestehenden Eintrag anhand seiner ID.

        Args:
            model (SQLAlchemy Model): Die SQLAlchemy-Modellklasse für die Tabelle.
            item_id (int): Die ID des zu aktualisierenden Eintrags.
            data (dict): Ein Dictionary mit den neuen Daten für den Eintrag.

        Returns:
            The updated model instance or None if not found or an error occurred.
        """
        session = self.SessionLocal()
        try:
            instance = session.query(model).filter(model.id == item_id).first()
            if instance:
                for key, value in data.items():
                    setattr(instance, key, value)
                session.commit()
                session.refresh(instance)
                return instance
            return None
        except Exception as e:
            session.rollback()
            print(f"Fehler beim Aktualisieren des Eintrags: {e}")
            return None
        finally:
            session.close()

    def delete(self, model, item_id):
        """
        Löscht einen Eintrag anhand seiner ID.

        Args:
            model (SQLAlchemy Model): Die SQLAlchemy-Modellklasse für die Tabelle.
            item_id (int): Die ID des zu löschenden Eintrags.

        Returns:
            True if the entry was successfully deleted, False otherwise.
        """
        session = self.SessionLocal()
        try:
            instance = session.query(model).filter(model.id == item_id).first()
            if instance:
                session.delete(instance)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Fehler beim Löschen des Eintrags: {e}")
            return False
        finally:
            session.close()

    def query(self, model, filters=None, order_by=None, limit=None, offset=None):
        """
        Führt eine allgemeine Abfrage auf der Datenbank durch.

        Args:
            model (SQLAlchemy Model): Die SQLAlchemy-Modellklasse für die Tabelle.
            filters (list, optional): Eine Liste von SQLAlchemy-Filterbedingungen. Defaults to None.
            order_by (SQLAlchemy Column, optional): Eine SQLAlchemy-Spalte zum Sortieren der Ergebnisse. Defaults to None.
            limit (int, optional): Die maximale Anzahl der zurückzugebenden Ergebnisse. Defaults to None.
            offset (int, optional): Der Offset für die zurückzugebenden Ergebnisse. Defaults to None.

        Returns:
            A list of model instances matching the query.
        """
        session = self.SessionLocal()
        try:
            query = session.query(model)
            if filters:
                query = query.filter(*filters)
            if order_by:
                query = query.order_by(order_by)
            if limit is not None:
                query = query.limit(limit)
            if offset is not None:
                query = query.offset(offset)
            return query.all()
        finally:
            session.close()


if __name__ == "__main__":
    # Beispielhafte Verwendung der Klasse

    # 1. Definiere ein SQLAlchemy-Modell
    class User(BaseModel):
        __tablename__ = "users"
        name = Column(String)
        email = Column(String, unique=True)

        def __repr__(self):
            return f"<User(id={self.id}, name='{self.name}', email='{self.email}')>"

    # 2. Initialisiere den DatabaseHandler mit einer Datenbank-Connection-String
    db_connection = "sqlite:///./mydatabase.db"  # Beispiel für eine SQLite-Datenbank
    db_handler = DatabaseHandler(db_connection)

    # 3. CRUD-Operationen

    # Create
    new_user_data = {"name": "Alice", "email": "alice@example.com"}
    new_user = db_handler.create(User, new_user_data)
    if new_user:
        print(f"Benutzer erstellt: {new_user}")

    new_user_data_2 = {"name": "Bob", "email": "bob@example.com"}
    new_user_2 = db_handler.create(User, new_user_data_2)
    if new_user_2:
        print(f"Benutzer erstellt: {new_user_2}")

    # Read
    user_alice = db_handler.read(User, 1)
    if user_alice:
        print(f"Gelesener Benutzer (ID 1): {user_alice}")

    user_not_found = db_handler.read(User, 99)
    if not user_not_found:
        print("Benutzer mit ID 99 nicht gefunden.")

    # Update
    updated_user_data = {"name": "Alice Smith", "email": "alice.smith@example.com"}
    updated_user = db_handler.update(User, 1, updated_user_data)
    if updated_user:
        print(f"Benutzer aktualisiert (ID 1): {updated_user}")

    # Query
    all_users = db_handler.query(User)
    print("Alle Benutzer:")
    for user in all_users:
        print(user)

    # Query mit Filter
    filtered_users = db_handler.query(User, filters=[User.name.like("%Alice%")])
    print("Benutzer mit 'Alice' im Namen:")
    for user in filtered_users:
        print(user)

    # Delete
    deleted = db_handler.delete(User, 2)
    if deleted:
        print("Benutzer mit ID 2 gelöscht.")
    else:
        print("Benutzer mit ID 2 nicht gefunden oder konnte nicht gelöscht werden.")

    # Query nach dem Löschen
    remaining_users = db_handler.query(User)
    print("Verbleibende Benutzer:")
    for user in remaining_users:
        print(user)
