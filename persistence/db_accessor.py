import psycopg
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseAccessor:
    def __init__(self, host: str = "localhost", database: str = "vivekanandvivek", 
                 user: str = "vivekanandvivek", password: str = "", port: int = 5432):
        self.connection_params = {
            "host": host,
            "dbname": database,
            "user": user,
            "password": password,
            "port": port
        }
        self.conn = None
        self._create_table_if_not_exists()
    
    def _get_connection(self):
        """Get a database connection."""
        try:
            if self.conn is None or self.conn.closed:
                self.conn = psycopg.connect(**self.connection_params)
            return self.conn
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def _create_table_if_not_exists(self):
        """Create the my_conversations table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS my_conversations (
            id SERIAL PRIMARY KEY,
            role VARCHAR(50) NOT NULL,
            response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(create_table_sql)
                conn.commit()
                logger.info("Table my_conversations created/verified successfully")
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise
    
    def insert_conversation(self, role: str, response: str) -> int:
        """Insert a new conversation entry and return the inserted ID."""
        insert_sql = """
        INSERT INTO my_conversations (role, response, timestamp)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(insert_sql, (role, response, datetime.now()))
                inserted_id = cursor.fetchone()[0]
                conn.commit()
                logger.info(f"Inserted conversation with ID: {inserted_id}")
                return inserted_id
        except Exception as e:
            logger.error(f"Error inserting conversation: {e}")
            if conn:
                conn.rollback()
            raise
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history, optionally limited to recent messages."""
        if limit:
            select_sql = """
            SELECT id, role, response, timestamp
            FROM my_conversations
            ORDER BY timestamp ASC
            LIMIT %s;
            """
            params = (limit,)
        else:
            select_sql = """
            SELECT id, role, response, timestamp
            FROM my_conversations
            ORDER BY timestamp ASC;
            """
            params = ()
        
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(select_sql, params)
                results = cursor.fetchall()
                # Convert to list of dictionaries
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in results]
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    
    
    
    def get_conversation_count(self) -> int:
        """Get total count of conversation entries."""
        count_sql = "SELECT COUNT(*) FROM my_conversations;"
        
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(count_sql)
                count = cursor.fetchone()[0]
                return count
        except Exception as e:
            logger.error(f"Error getting conversation count: {e}")
            return 0
    
    def clear_conversation_history(self) -> int:
        """Clear all conversation history and return the number of deleted entries."""
        delete_sql = "DELETE FROM my_conversations;"
        
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(delete_sql)
                deleted_count = cursor.rowcount
                conn.commit()
                logger.info(f"Cleared {deleted_count} conversation entries")
                return deleted_count
        except Exception as e:
            logger.error(f"Error clearing conversation history: {e}")
            if conn:
                conn.rollback()
            return 0
    
    def close_connection(self):
        """Close the database connection."""
        try:
            if self.conn and not self.conn.closed:
                self.conn.close()
                logger.info("Database connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
