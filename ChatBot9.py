import os
import tkinter as tk
import tkinter.ttk as ttk
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import OpenAI
from langchain_cohere import ChatCohere

# # Set OpenAI API key
# os.environ['OPENAI_API_KEY'] = "sk-proj-sdWtibOmo0eSvaDroqqET3BlbkFJmsqtJuVbpU3T0SeRGXnj"
# Set up Cohere API key
os.environ['COHERE_API_KEY'] = "w9BCnpVENCLMBfaUkuSH1hEGrWNKexfD4N9aq3X3"

connection_string = "mssql+pyodbc://sa:Pratham72@localhost/Trial1?driver=ODBC+Driver+17+for+SQL+Server"

# Define the connection string for Microsoft SQL Server
# server_name = 'localhost'
# database_name = 'Trial1'
# connection_string = f'mssql+pyodbc://@{server_name}/{database_name}?driver=ODBC+Driver+17+for+SQL+Server'

# Connect to the database using SQLAlchemy engine
engine = create_engine(connection_string)

# Create the SQLDatabase object
db = SQLDatabase(engine)

# # Create the agent executor for OpenAI
# toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))
# agent_executor = create_sql_agent(
#     llm=OpenAI(temperature=0),
#     toolkit=toolkit,
#     verbose=True
# )

# Create the agent executor
toolkit = SQLDatabaseToolkit(db=db, llm=ChatCohere(model="command-r", temperature=0))
agent_executor = create_sql_agent(
    llm=ChatCohere(temperature=0),
    toolkit=toolkit,
    verbose=True
)

# Create the UI window
root = tk.Tk()
root.title("Chat with your Microsoft SQL Data")

# Create the text entry widget
entry = ttk.Entry(root, font=("Times New Roman", 14))
entry.pack(padx=20, pady=20, fill=tk.X)

# Create the button callback
def on_click():
    # Get the query text from the entry widget
    query = entry.get()

    # Run the query using the agent executor
    result = agent_executor.run(query)

    # Display the result in the text widget
    text.delete("1.0", tk.END)
    text.insert(tk.END, result)

# Bind the Enter key to the on_click function
root.bind('<Return>', lambda event: on_click())

# Create the button widget
button = ttk.Button(root, text="Ask", command=on_click)
button.pack(padx=20, pady=20)

# Create the text widget to display the result
text = tk.Text(root, height=10, width=80, font=("Times New Roman", 14))
text.pack(padx=20, pady=20)

# Start the UI event loop
root.mainloop()