import google.generativeai as genai
import json
import re
import math
import os

# --- Configuration ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-3.0-flash')

# --- System Prompt ---
system_prompt = """You are a helpful AI agent that can use tools to answer questions accurately.

You have access to the following tools:

1. calculate(expression: str) -> str
   Evaluate a mathematical expression. Example: calculate("2**10")

2. get_weather(city: str) -> str
   Get the current weather for a city. Example: get_weather("Mumbai")

3. search_notes(query: str) -> str
   Search through user's notes. Example: search_notes("meeting agenda")

You must respond in ONE of these two JSON formats:

If you need to use a tool:
{"tool_name": "<name>", "tool_arguments": {"<arg_name>": "<value>"}}

If you have the final answer:
{"answer": "<your final answer>"}

IMPORTANT: 
- Respond with ONLY the JSON. No other text.
- Use tools when you need real data or calculations.
- After receiving a tool result, either use another tool or provide your final answer.
"""

# --- Tools ---
def calculate(expression: str) -> str:
    try:
        allowed = {"math": math, "abs": abs, "round": round, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return json.dumps({"result": str(result)})
    except Exception as e:
        return json.dumps({"error": str(e)})

def get_weather(city: str) -> str:
    weather_data = {
        "Mumbai": "32°C, Humid, Partly Cloudy",
        "Delhi": "28°C, Clear Sky",
        "London": "15°C, Rainy",
        "New York": "22°C, Sunny",
        "Tokyo": "26°C, Windy",
    }
    weather = weather_data.get(city, f"Weather data not available for {city}")
    return json.dumps({"weather": weather})

def search_notes(query: str) -> str:
    notes = [
        {"title": "Meeting Agenda", "content": "Discuss Q3 targets, review agent architecture"},
        {"title": "Shopping List", "content": "Milk, eggs, bread, coffee"},
        {"title": "Project Ideas", "content": "Build a stock monitoring agent, voice assistant"},
    ]
    results = [n for n in notes if query.lower() in n["title"].lower() or query.lower() in n["content"].lower()]
    return json.dumps({"results": results if results else "No notes found"})

tools = {
    "calculate": calculate,
    "get_weather": get_weather,
    "search_notes": search_notes,
}

# --- Response Parser ---
def parse_llm_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(lines).strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"Could not parse: {text}")

# --- The Agent Loop ---
def run_agent(user_query: str, max_iterations: int = 5):
    print(f"\n{'='*60}")
    print(f"User: {user_query}")
    print(f"{'='*60}")

    # Build initial conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Build the prompt from message history
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += msg["content"] + "\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n\n"
            elif msg["role"] == "tool":
                prompt += f"Tool Result: {msg['content']}\n\n"

        # Call the LLM
        response = model.generate_content(prompt)
        response_text = response.text
        print(f"LLM Response: {response_text}")

        # Parse the response
        try:
            parsed = parse_llm_response(response_text)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Parse error: {e}")
            print("Asking LLM to try again...")
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": "Please respond with valid JSON only."})
            continue

        # Check if it's a final answer
        if "answer" in parsed:
            print(f"\n{'='*60}")
            print(f"Agent Answer: {parsed['answer']}")
            print(f"{'='*60}")
            return parsed["answer"]

        # It's a tool call
        if "tool_name" in parsed:
            tool_name = parsed["tool_name"]
            tool_args = parsed.get("tool_arguments", {})

            print(f"Calling tool: {tool_name}({tool_args})")

            if tool_name not in tools:
                print(f"Unknown tool: {tool_name}")
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "tool", "content": json.dumps({"error": f"Unknown tool: {tool_name}"})})
                continue

            # Execute the tool
            tool_result = tools[tool_name](**tool_args)
            print(f"Tool Result: {tool_result}")

            # Add to conversation history
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "tool", "content": tool_result})

    print("\nMax iterations reached. Agent could not complete the task.")
    return None


# --- Run it! ---
if __name__ == "__main__":
    # Test 1: Simple tool call
    run_agent("What is the weather in Mumbai?")

    # Test 2: Calculation
    run_agent("What is 2 raised to the power of 10, plus the square root of 144?")

    # Test 3: Multi-step reasoning
    run_agent("Search my notes for project ideas, then tell me the weather in Tokyo so I can decide if I should work from a cafe there.")

    # Test 4: Something that needs multiple tools
    run_agent("Calculate the sum of exponential values of the first 6 Fibonacci numbers")