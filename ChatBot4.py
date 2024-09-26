import os
import uuid
import pyodbc
from typing import List, Dict
import cohere

# Assuming 'cohere' is the package name for Cohere API
co = cohere.Client("D14bT4Bm9SoiXE5ioVryf2DGOyIw1yjm1ccR0giQ")

class DatabaseQuery:
    def __init__(self, server: str, database: str, username: str, password: str):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.connection = self.create_connection()

    def create_connection(self):
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}"
        try:
            connection = pyodbc.connect(conn_str)
            print("Database connection established.")
            return connection
        except pyodbc.Error as e:
            print(f"Error connecting to SQL Server: {e}")
            raise
        # return pyodbc.connect(conn_str)

    def execute_query(self, query: str):
        """
        Executes a SQL query on the connected MSSQL database.

        Parameters:
        query (str): The SQL query to execute.

        Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the query results.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            columns = [column[0] for column in cursor.description]
            result = [dict(zip(columns, row)) for row in cursor.fetchall()]
            # result = cursor.fetchall()
            cursor.close()
            return result
        except pyodbc.Error as e:
            print(f"Error executing query: {e}")
            raise    

class Vectorstore:
    def __init__(self, database_query: DatabaseQuery):
        self.database_query = database_query

    def retrieve_from_database(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves data from MSSQL database based on the given query.

        Parameters:
        query (str): The SQL query to retrieve data from the database.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved data rows.
        """
        print(f"Retrieving data from database with query: {query}")
        results = self.database_query.execute_query(query)

        # Format results as needed (assuming each row is a dictionary with column names as keys)
        formatted_results = []
        for row in results:
            formatted_results.append(dict(row))

        return formatted_results
    
class Chatbot:
    def __init__(self, vectorstore: Vectorstore):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())

    def run(self):
        while True:
            message = input("User: ")

            if message.lower() == "quit":
                print("Ending chat.")
                break
            else:
                print(f"User: {message}")

            # For testing, use a predefined query instead of user input
            query = "SELECT TOP (1000) [user_id], [username], [email], [password] FROM [Trial1].[dbo].[Users]"

            # Query database based on user message
            try:
                db_results = self.vectorstore.retrieve_from_database(query)
            except Exception as e:
                print(f"Error retrieving from database: {e}")
                continue

            # Prepare database results for chatbot response
            documents = []
            for result in db_results:
                documents.append({
                    "user_id": result["user_id"],  # Adjust as per your database schema
                    "username": result["username"],  # Adjust as per your database schema
                    "email": result["email"],  # Adjust as per your database schema
                    "password": result["password"],  # Adjust as per your database schema
                    # Add other fields as necessary
                })

            try:
                # Use Cohere to process user message and retrieve relevant documents
                response = co.chat_stream(
                    message=message,
                    model="command-r",
                    documents=documents,
                    conversation_id=self.conversation_id,
                )
            except Exception as e:
                print(f"Error during Cohere chat_stream: {e}")
                continue

            # Print chatbot response
            print("\nChatbot:")
            citations = []
            cited_documents = []

            for event in response:
                if event.event_type == "text-generation":
                    print(event.text, end="")
                elif event.event_type == "citation-generation":
                    citations.extend(event.citations)
                elif event.event_type == "search-results":
                    cited_documents = event.documents

            if citations:
                print("\n\nCITATIONS:")
                for citation in citations:
                    print(citation)

                print("\nDOCUMENTS:")
                for document in cited_documents:
                    print(document)

            print(f"\n{'-'*100}\n")

if __name__ == "__main__":
    # Replace these with your MSSQL server details
    server = 'localhost'
    database = 'Trial1'
    username = 'sa'
    password = 'Pratham72'

    # Create an instance of DatabaseQuery
    database_query = DatabaseQuery(server, database, username, password)

    # Create an instance of Vectorstore
    vectorstore = Vectorstore(database_query)

    # Create an instance of Chatbot
    chatbot = Chatbot(vectorstore)

    # Run the chatbot
    chatbot.run()

    # Close database connection
    database_query.connection.close()

# class Chatbot:
#     def __init__(self, vectorstore: Vectorstore):
#         self.vectorstore = vectorstore
#         self.conversation_id = str(uuid.uuid4())

#     def run(self):
#         while True:
#             message = input("User: ")

#             if message.lower() == "quit":
#                 print("Ending chat.")
#                 break
#             else:
#                 print(f"User: {message}")

#             # Query database based on user message
#             db_results = self.vectorstore.retrieve_from_database(message)

#             # Prepare database results for chatbot response
#             documents = []
#             for result in db_results:
#                 documents.append({
#                     "user_id": result["user_id"],  # Adjust as per your database schema
#                     "username": result["username"],  # Adjust as per your database schema
#                     # Add other fields as necessary
#                 })

#             # Use Cohere to process user message and retrieve relevant documents
#             response = co.chat_stream(
#                 message=message,
#                 model="command-r",
#                 documents=documents,
#                 conversation_id=self.conversation_id,
#             )

#             # Print chatbot response
#             print("\nChatbot:")
#             citations = []
#             cited_documents = []

#             for event in response:
#                 if event.event_type == "text-generation":
#                     print(event.text, end="")
#                 elif event.event_type == "citation-generation":
#                     citations.extend(event.citations)
#                 elif event.event_type == "search-results":
#                     cited_documents = event.documents

#             if citations:
#                 print("\n\nCITATIONS:")
#                 for citation in citations:
#                     print(citation)

#                 print("\nDOCUMENTS:")
#                 for document in cited_documents:
#                     print(document)

#             print(f"\n{'-'*100}\n")

# if __name__ == "__main__":
#     # Replace these with your MSSQL server details
#     server = 'localhost'
#     database = 'Trial1'
#     username = 'sa'
#     password = 'Pratham72'

#     # Create an instance of DatabaseQuery
#     database_query = DatabaseQuery(server, database, username, password)

#     # Create an instance of Vectorstore
#     vectorstore = Vectorstore(database_query)

#     # Create an instance of Chatbot
#     chatbot = Chatbot(vectorstore)

#     # Run the chatbot
#     chatbot.run()

#     # Close database connection
#     database_query.connection.close()
