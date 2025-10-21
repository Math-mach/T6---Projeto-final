# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    predictions = relationship('Prediction', backref='user', lazy=True)

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    upload_timestamp = Column(DateTime, nullable=False, default=func.now())
    jogador_id = Column(String, nullable=False)
    pred_t1 = Column(Float, nullable=False)
    pred_t2 = Column(Float, nullable=False)
    pred_t3 = Column(Float, nullable=False)