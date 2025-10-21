# crud.py
from sqlalchemy.orm import Session
import models
import auth

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user_schema: models.User):
    hashed_password = auth.get_password_hash(user_schema.password)
    db_user = models.User(username=user_schema.username, password_hash=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user