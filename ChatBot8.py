import os
import tkinter as tk
import tkinter.ttk as ttk
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
import cohere

class CohereLLM:
    def __init__(self, api_key, temperature=0.5):
        self.client = cohere.Client(api_key)
        self.temperature = temperature

    def __call__(self, prompt):
        response = self.client.generate(
            model='command-xlarge-nightly',  # Specify the model size
            prompt=prompt,
            max_tokens=100,  # Adjust max tokens as needed
            temperature=self.temperature
        )
        return response.generations[0].text.strip()

# Set Cohere API key
cohere_api_key = "D14bT4Bm9SoiXE5ioVryf2DGOyIw1yjm1ccR0giQ"
co = CohereLLM(api_key=cohere_api_key, temperature=0.5)

connection_string = "mssql+pyodbc://sa:Pratham72@localhost/Trial1?driver=ODBC+Driver+17+for+SQL+Server"

# Connect to the database using SQLAlchemy engine
engine = create_engine(connection_string)

# Create the SQLDatabase object
db = SQLDatabase(engine)

# Create the agent executor
toolkit = SQLDatabaseToolkit(db=db, llm=co)
agent_executor = create_sql_agent(
    llm=co,
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
