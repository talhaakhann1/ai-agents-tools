from dotenv import load_dotenv
import os
import requests

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from tavily import TavilyClient
from rich import print

load_dotenv()


# 1. Weather tool
@tool
def get_Weather(city: str) -> str:
    """Get the current weather of a city in Pakistan."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Error: OPENWEATHER_API_KEY is missing."

    url = "https://api.openweathermap.org/data/2.5/weather"

    try:
        response = requests.get(
            url,
            params={
                "q": f"{city},PK",
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
        return f"Weather API error: {error}"

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


# 2. Tavily news tool
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY is missing from your .env file.")

tavily_client = TavilyClient(api_key=tavily_api_key)


@tool
def get_news(city: str) -> str:
    """Search for recent news about a city."""

    try:
        response = tavily_client.search(
            query=f"latest news in {city}, Pakistan",
            search_depth="basic",
            max_results=3,
        )
    except Exception as error:
        return f"News search error: {error}"

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}."

    news_list = []

    for result in results:
        title = result.get("title", "No title")
        url = result.get("url", "")
        snippet = result.get("content", "")

        news_list.append(
            f"- {title}\n"
            f"  Link: {url}\n"
            f"  Summary: {snippet[:200]}"
        )

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# 3. Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

tools = {
    "get_Weather": get_Weather,
    "get_news": get_news,
}

llm_tools = llm.bind_tools([get_Weather, get_news])


# 4. Agent conversation loop
messages = []

print("[bold green]City Intelligence System[/bold green]")
print("Ask about weather or city news.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    if not user_input:
        continue

    messages.append(HumanMessage(content=user_input))

    while True:
        result = llm_tools.invoke(messages)

        # No tool calls: the model has a normal answer.
        if not result.tool_calls:
            messages.append(result)
            print(f"\n[bold cyan]Assistant:[/bold cyan] {result.content}\n")
            break

        # Save the assistant message containing tool-call requests.
        messages.append(result)

        # Execute each requested tool and return its result.
        for tool_call in result.tool_calls:
            tool_name = tool_call["name"]
            tool_id = tool_call["id"]
            tool_args = tool_call["args"]

            print(f"\nAgent requests tool: {tool_name}")
            print(f"Arguments: {tool_args}")

            confirm = input("Approve this tool call? (yes/no): ").strip().lower()

            if confirm not in ("yes", "y"):
                tool_output = "Tool call denied by the user."
                print("Tool call denied.")
            else:
                tool_function = tools.get(tool_name)

                if tool_function is None:
                    tool_output = f"Error: Unknown tool '{tool_name}'."
                else:
                    try:
                        tool_output = str(tool_function.invoke(tool_args))
                    except Exception as error:
                        tool_output = f"Tool execution error: {error}"

                print(f"\nTool result:\n{tool_output}\n")

            messages.append(
                ToolMessage(
                    content=tool_output,
                    tool_call_id=tool_id,
                )
            )

        # Loop back to the LLM with the tool results.
        # It may request another tool or provide the final answer.


