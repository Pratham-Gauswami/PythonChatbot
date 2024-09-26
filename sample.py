import os
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_cohere import ChatCohere
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# Set up Cohere API key
os.environ['COHERE_API_KEY'] = "w9BCnpVENCLMBfaUkuSH1hEGrWNKexfD4N9aq3X3"

# # Set up connection to the SQL Server database
# connection_string = "mssql+pyodbc://sa:Pratham72@localhost/Trial1?driver=ODBC+Driver+17+for+SQL+Server"
# engine = create_engine(connection_string)
# db = SQLDatabase(engine)

db = ("mssql+pyodbc://sa:Pratham72@Trial1")
db_engine = create_engine(db)
# print(db.get_tables_names())

# Initialize the ChatCohere language model
llm = ChatCohere(model="command-r", temperature=0)

# Define tools for writing and executing queries
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)

# Combine the tools into a chain
chain = write_query | execute_query

# Invoke the chain with the question
response = chain.invoke({"question": "How many Users are there?"})

# Print the response
print(response)
