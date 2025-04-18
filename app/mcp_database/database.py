import os
from pathlib import Path  # Use pathlib for easier path manipulation
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime


# Use the modern way to define Base
class Base(DeclarativeBase):
    pass


class BaseModel(Base):
    __abstract__ = True
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatabaseHandler:
    DB_SUBDIR_FROM_ROOT = Path("data") / "database"

    def __init__(self, db_filename: str = "default_database.db"):
        """
        Initialisiert den DatabaseHandler und stellt sicher, dass die Datenbank
        im Unterverzeichnis data/database relativ zum Projekt-Root erstellt wird.

        Args:
            db_filename (str): Der Name der Datenbankdatei (z.B. 'mydatabase.db').
                                Defaults to "default_database.db".
        """
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        self.db_directory = project_root / self.DB_SUBDIR_FROM_ROOT
        self.db_path = self.db_directory / db_filename
        self.db_directory.mkdir(parents=True, exist_ok=True)
        db_connection_string = f"sqlite:///{self.db_path.resolve()}"
        print(
            f"Datenbank wird initialisiert unter: {self.db_path.resolve()}"
        )  # Info message

        self.engine = create_engine(db_connection_string)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create(self, model, data):
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
        session = self.SessionLocal()
        try:
            return session.get(model, item_id)
        finally:
            session.close()

    def update(self, model, item_id, data):
        session = self.SessionLocal()
        try:
            instance = session.get(model, item_id)
            if instance:
                for key, value in data.items():
                    if key not in ["id", "created_at"]:
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
        session = self.SessionLocal()
        try:
            instance = session.get(model, item_id)
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
        session = self.SessionLocal()
        try:
            query = session.query(model)
            if filters:
                if not isinstance(filters, (list, tuple)):
                    filters = [filters]
                query = query.filter(*filters)
            if order_by is not None:
                if not isinstance(order_by, (list, tuple)):
                    order_by = [order_by]
                query = query.order_by(*order_by)
            if limit is not None:
                query = query.limit(limit)
            if offset is not None:
                query = query.offset(offset)
            return query.all()
        except Exception as e:
            print(f"Fehler bei der Abfrage: {e}")
            return []
        finally:
            session.close()


# --- Example Usage ---
if __name__ == "__main__":

    class User(BaseModel):
        __tablename__ = "users"
        name = Column(String)
        email = Column(String, unique=True)

        def __repr__(self):
            return f"<User(id={self.id}, name='{self.name}', email='{self.email}')>"

    db_handler = DatabaseHandler("main_database.db")

    print("\n--- Create ---")
    new_user_data = {"name": "Alice", "email": "alice@example.com"}
    new_user = db_handler.create(User, new_user_data)
    if new_user:
        print(f"Benutzer erstellt: {new_user}")

    new_user_data_2 = {"name": "Bob", "email": "bob@example.com"}
    new_user_2 = db_handler.create(User, new_user_data_2)
    if new_user_2:
        print(f"Benutzer erstellt: {new_user_2}")

    user_alice = db_handler.read(User, 1)
    if user_alice:
        print(f"Gelesener Benutzer (ID 1): {user_alice}")

    user_not_found = db_handler.read(User, 99)
    if not user_not_found:
        print("Benutzer mit ID 99 nicht gefunden.")

    updated_user_data = {"name": "Alice Smith", "email": "alice.smith@example.com"}
    updated_user = db_handler.update(User, 1, updated_user_data)
    if updated_user:
        print(f"Benutzer aktualisiert (ID 1): {updated_user}")

    all_users = db_handler.query(User)
    print("Alle Benutzer:")
    for user in all_users:
        print(user)

    filtered_users = db_handler.query(User, filters=[User.name.like("%Alice%")])
    print("Benutzer mit 'Alice' im Namen:")
    for user in filtered_users:
        print(user)

    deleted = db_handler.delete(User, 2)
    if deleted:
        print("Benutzer mit ID 2 gelöscht.")
    else:
        print("Benutzer mit ID 2 nicht gefunden oder konnte nicht gelöscht werden.")

    remaining_users = db_handler.query(User)
    print("Verbleibende Benutzer:")
    for user in remaining_users:
        print(user)
