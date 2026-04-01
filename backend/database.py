"""
Configuración de Base de Datos y Modelos SQLAlchemy

Este módulo define la estructura de la base de datos SQLite
utilizada para almacenar usuarios, alertas y métricas del modelo.
Implementa el patrón ORM para facilitar la interacción con los datos.

Tablas:
- usuarios: Gestión de usuarios y autenticación
- alertas: Registro de alertas generadas por el sistema
- metricas_modelo: Histórico de métricas de rendimiento del modelo
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from .config import settings

# Configuración del motor de base de datos
# check_same_thread=False es necesario para SQLite con FastAPI
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Fábrica de sesiones
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base declarativa para los modelos
Base = declarative_base()


class Usuario(Base):
    """
    Modelo de Usuario para autenticación y autorización.

    Almacena la información de los usuarios del sistema,
    incluyendo credenciales hasheadas y roles.

    Atributos:
        id: Identificador único del usuario
        username: Nombre de usuario para autenticación
        email: Correo electrónico (opcional)
        password_hash: Hash seguro de la contraseña (bcrypt)
        rol: Rol del usuario (admin, analista)
        fecha_creacion: Fecha de registro del usuario
    """
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    rol = Column(String(20), default="analista")
    fecha_creacion = Column(DateTime, default=datetime.utcnow)

    # Relaciones
    alertas = relationship("Alerta", back_populates="usuario")
    metricas = relationship("MetricaModelo", back_populates="usuario")
    reset_tokens = relationship("PasswordResetToken", back_populates="usuario")

    def __repr__(self):
        return f"<Usuario(id={self.id}, username='{self.username}', rol='{self.rol}')>"


class PasswordResetToken(Base):
    """
    Modelo de Token de Reseteo de Contraseña.

    Almacena tokens únicos y seguros para permitir que los usuarios
    reseteen sus contraseñas de forma segura mediante email.

    Atributos:
        id: Identificador único del token
        usuario_id: Usuario asociado al token
        token: Token único y seguro (hash)
        expiracion: Fecha y hora de expiración del token
        usado: Si el token ya fue utilizado
        fecha_creacion: Fecha de creación del token
    """
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    token = Column(String(255), unique=True, index=True, nullable=False)
    expiracion = Column(DateTime, nullable=False)
    usado = Column(Integer, default=0)  # 0 = no usado, 1 = usado
    fecha_creacion = Column(DateTime, default=datetime.utcnow)

    # Relación con usuario
    usuario = relationship("Usuario", back_populates="reset_tokens")

    def __repr__(self):
        return f"<PasswordResetToken(id={self.id}, usuario_id={self.usuario_id}, usado={self.usado})>"


class Alerta(Base):
    """
    Modelo de Alerta para registro de eventos críticos.

    Almacena las alertas generadas por el sistema cuando
    se detectan movimientos significativos en los activos
    financieros monitoreados.

    Atributos:
        id: Identificador único de la alerta
        usuario_id: Usuario asociado a la alerta
        ticker: Símbolo del activo financiero
        severidad: Nivel de severidad (info, advertencia, critica)
        mensaje: Descripción de la alerta
        senal: Señal cuantitativa del modelo
        recomendacion: Recomendación generada por el agente
        variacion_pct: Variación porcentual esperada
        fecha: Fecha y hora de creación de la alerta
    """
    __tablename__ = "alertas"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=True)
    ticker = Column(String(20), nullable=False, index=True)
    tipo = Column(String(20), nullable=False)  # info, warning, critical
    mensaje = Column(Text, nullable=False)
    variacion_pct = Column(Float, nullable=True)
    precio_actual = Column(Float, nullable=True)
    precio_predicho = Column(Float, nullable=True)
    leida = Column(Integer, default=0)  # 0 = no leída, 1 = leída (SQLite no tiene boolean nativo)
    fecha_creacion = Column(DateTime, default=datetime.utcnow, index=True)

    # Relación con usuario
    usuario = relationship("Usuario", back_populates="alertas")

    def __repr__(self):
        return f"<Alerta(id={self.id}, ticker='{self.ticker}', tipo='{self.tipo}')>"

    def to_dict(self):
        """Convierte la alerta a diccionario para serialización JSON."""
        return {
            "id": self.id,
            "usuario_id": self.usuario_id,
            "ticker": self.ticker,
            "tipo": self.tipo,
            "mensaje": self.mensaje,
            "variacion_pct": self.variacion_pct,
            "precio_actual": self.precio_actual,
            "precio_predicho": self.precio_predicho,
            "leida": self.leida,
            "fecha_creacion": self.fecha_creacion.isoformat() if self.fecha_creacion else None
        }


class MetricaModelo(Base):
    """
    Modelo para registro de métricas de rendimiento.

    Almacena el histórico de métricas de evaluación del modelo
    predictivo, permitiendo monitorear su evolución temporal
    y detectar posibles degradaciones.

    Atributos:
        id: Identificador único del registro
        usuario_id: Usuario que ejecutó la predicción
        ticker: Símbolo del activo analizado
        modelo: Nombre del modelo utilizado
        accuracy: Exactitud general del clasificador
        precision: Precisión de clase positiva (SUBIDA)
        recall: Recall de clase positiva (SUBIDA)
        f1: F1-Score (balance precision-recall)
        auc: Área bajo la curva ROC (AUC-ROC)
        fecha: Fecha y hora del registro
    """
    __tablename__ = "metricas_modelo"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=True)
    ticker = Column(String(20), nullable=False, index=True)
    modelo = Column(String(50), default="ensemble_clasificador")
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    auc = Column(Float, nullable=True)
    fecha = Column(DateTime, default=datetime.utcnow)

    # Relación con usuario
    usuario = relationship("Usuario", back_populates="metricas")

    def __repr__(self):
        return f"<MetricaModelo(id={self.id}, ticker='{self.ticker}', accuracy={self.accuracy})>"


def get_db():
    """
    Generador de sesiones de base de datos.

    Utilizado como dependencia en FastAPI para inyectar
    la sesión de base de datos en los endpoints.

    Yields:
        Session: Sesión de SQLAlchemy
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Inicializa la base de datos creando todas las tablas.

    Debe ejecutarse al iniciar la aplicación para asegurar
    que el esquema de base de datos existe.
    """
    Base.metadata.create_all(bind=engine)
