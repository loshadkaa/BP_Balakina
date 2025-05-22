from flask_sqlalchemy import SQLAlchemy
from datetime import datetime  # Add this import at the top

db = SQLAlchemy()

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    surname = db.Column(db.String(100), nullable=False)
    birth_date = db.Column(db.Date, nullable=False)
    notes = db.Column(db.Text)
    # Add other fields as needed

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'))
    description = db.Column(db.Text)
    is_test = db.Column(db.Boolean, default=False)
    model_used = db.Column(db.String(50))
    # Add other fields as needed

class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    version = db.Column(db.String(50))
    # Add other fields as needed