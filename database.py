from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

# The database is just a local file called study_buddy.db
engine = create_engine(
    "sqlite:///study_buddy.db",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# One row per user account
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    token = Column(String, nullable=True)


# One row per conversation (a single chat thread)
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, default="New chat")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# One row per message, linked to both the user and the conversation
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)        # "user" or "bot"
    content = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# Create the tables if they don't exist yet
Base.metadata.create_all(bind=engine)
