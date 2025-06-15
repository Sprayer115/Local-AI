import os

# Set OpenAI API configuration BEFORE importing praisonaiagents
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
os.environ["OPENAI_API_KEY"] = "not-needed"

from praisonaiagents import Agent, MCP,  Task, PraisonAIAgents

python_path = os.getenv("PYTHON_PATH", "python")  # or full path like "/usr/bin/python3"
server_path = os.getenv("WEATHER_SERVER_PATH", "/home/simon/entwicklung/Local-AI/mcp-services/weather-service/src/weather_server.py")  # path to your weather server

 # Create the agent with Ollama
weather_agent = Agent(
    name="Weather Assistant",
    role="Weather assistant",
    goal="Provide accurate and timely weather information for various cities",
    instructions="""You are a helpful weather assistant that can provide current weather information, 
    forecasts, and weather comparisons for different cities. Use the available weather tools to answer 
    user questions about weather conditions. You can:
    
    - Get current weather for cities
    - Get hourly forecasts 
    - Compare weather between two cities
    - Use both mock data and real API data (when API key is provided)
    - set use_real_api True to use real API data all the time
    
    Always use the appropriate weather tools when users ask about weather information.
    """,
    
    llm="ollama/llama3.2",  # Using Ollama with llama3.2
    
    # MCP server connection - adjust paths as needed
    tools=MCP(f"{python_path} {server_path}"),
    verbose=True
)


# Example queries to test
test_queries = [
    "Use the actual weather API to get the current weather in New York City.",
]

for query in test_queries:
    print(f"\n🔍 Query: {query}")
    print("-" * 30)
    try:
        response = weather_agent.start(prompt=query)
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    print("-" * 50)