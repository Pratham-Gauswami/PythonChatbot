from langchain_community.utilities import SQLDatabase
import os
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.agent_toolkits import create_sql_agent

os.environ['COHERE_API_KEY'] = "w9BCnpVENCLMBfaUkuSH1hEGrWNKexfD4N9aq3X3"

llm = ChatCohere(model="command-r", temperature=0)
db = SQLDatabase.from_uri("mssql+pyodbc://sa:Pratham72@localhost/Trial1?driver=ODBC+Driver+17+for+SQL+Server")


answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)

answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

# agent_executor = create_sql_agent(llm, db=db, agent_type="tool-caling", verbose=True)

# chain = write_query | execute_query
response = chain.invoke({"question": "can you run it like that"})

# reply = agent_executor.invoke(
#     {
#         "input": "How many users are there"
#     }
# )

print(db.get_usable_table_names())
print("\n")
print(db.run("SELECT * FROM Users;"))
print("\n")
print(response)