import pyodbc

def create_connection(server, database, username, password):
    connection_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"TrustServerCertificate=True;"  # Add this line to trust the server certificate
    )
    connection = None
    try:
        connection = pyodbc.connect(connection_string)
        print("Connection to MSSQL DB successful")
    except pyodbc.Error as e:
        print(f"The error '{e}' occurred")
    
    return connection

# Create a connection using the function
connection = create_connection("localhost", "Trial1", "sa", "Pratham72")
