from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.chat_models import ChatCohere
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import os
import json

# load the cohere api key
os.environ["COHERE_API_KEY"] = "D14bT4Bm9SoiXE5ioVryf2DGOyIw1yjm1ccR0giQ"

DB_NAME='Chinook.db'
MODEL="command-r-plus"
llm = ChatCohere(model=MODEL, temperature=0.1,verbose=True)
db = SQLDatabase.from_uri(f"sqlite:///{DB_NAME}")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
context = toolkit.get_context()
tools = toolkit.get_tools()

print('**List of pre-defined Langchain Tools**')
print([tool.name for tool in tools])

# define the prompt template
prompt = ChatPromptTemplate.from_template("{input}")
# instantiate the ReAct agent
agent = create_cohere_react_agent(
   llm=llm,
   tools=tools,
   prompt=prompt,
)
agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               verbose=True,
                               return_intermediate_steps=True
                    )

output=agent_executor.invoke({
   "input": 'what tables are available?',
})

print(output['output'])

output=agent_executor.invoke({
   "input": 'show the first row of the Playlist and Genre tables?',
})


### CHATBOT ## Pre trained
# import cohere
# import uuid


# co = cohere.Client("D14bT4Bm9SoiXE5ioVryf2DGOyIw1yjm1ccR0giQ") # Your Cohere API key

# # Create a conversation ID
# conversation_id = str(uuid.uuid4())

# # Define the preamble
# preamble = "You are an expert public speaking coach"

# print('Starting the chat. Type "quit" to end.\n')

# while True:

#     # User message
#     message = input("User: ")

#     # Typing "quit" ends the conversation
#     if message.lower() == 'quit':
#         print("Ending chat.")
#         break

#     # Chatbot response
#     stream = co.chat_stream(message=message,
#                             model="command-r-plus",
#                             preamble=preamble,
#                             conversation_id=conversation_id)

#     print("Chatbot: ", end='')

#     for event in stream:
#         if event.event_type == "text-generation":
#             print(event.text, end='')
#         if event.event_type == "stream-end":
#             chat_history = event.response.chat_history

#     print(f"\n{'-'*100}\n")

