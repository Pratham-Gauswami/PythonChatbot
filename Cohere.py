import os
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_cohere import ChatCohere
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

server = 'localhost'
database = 'Trial1'
username = 'sa'
password = 'Pratham72'

# Set up connection to the SQL Server database
# connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
connection_string = "mssql+pyodbc://sa:Pratham72@localhost/Trial1?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(connection_string)
db = SQLDatabase(engine)

# Set up Cohere API key
os.environ['COHERE_API_KEY'] = "w9BCnpVENCLMBfaUkuSH1hEGrWNKexfD4N9aq3X3"

# Initialize the ChatCohere language model
llm = ChatCohere(model="command-r", temperature=0)

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query
# answer_prompt = PromptTemplate.from_template(
#     """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

# Question: {question}
# SQL Query: {query}
# SQL Result: {result}
# Answer: """
# )

# chain = (
#     RunnablePassthrough.assign(query=write_query).assign(
#         result=itemgetter("query") | execute_query
#     )
#     | answer_prompt
#     | llm
#     | StrOutputParser()
# )

response = chain.invoke({"question": "How many Users are there in the table?"})
print(response)