import pymssql
from langchain_cohere import ChatCohere
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase


# Set up pymssql connection parameters
server = 'localhost'
database = 'Trial1'
username = 'sa'
password = 'Pratham72'

# Set up Langchain components
llm = ChatCohere(model="command-r", temperature=0)
conn = pymssql.connect(server, username, password, database)
db = SQLDatabase(conn)
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)

# Example usage in Langchain workflow
try:
    # Process user input and generate SQL query using Langchain
    question = "How many users are there?"
    chain = write_query | execute_query
    response = chain.invoke({"question": question})
    print(f"Langchain Response: {response}")

    # Example using pymssql directly
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM Users")
    result = cursor.fetchone()
    print(f"Total number of users: {result[0]}")

finally:
    # Clean up resources
    cursor.close()
    conn.close()
