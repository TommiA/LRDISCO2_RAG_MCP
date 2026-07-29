import asyncio
import os
import sys
import argparse
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model

def init_LM_studio_model(base_url):
    model = init_chat_model(
        model="qwen/qwen3-vl-8b",
        model_provider="openai",
        base_url=base_url,
        api_key="not-needed",
    )
    return model

async def make_query(model, tools, user_query):
    messages = [
        SystemMessage(
            content=(
                "You are a helpful Land Rover Discovery 2 mechanic assistant. "
                "Use the available tools when needed. "
                "Stay concise. If you don't know, say so."
            )
        ),
        HumanMessage(content=user_query),
    ]

    while True:
        response = await model.ainvoke(messages)
        messages.append(response)
        # Normal answer
        if not response.tool_calls:
            return response.content

        # Handle MCP tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            selected_tool = next(
                tool for tool in tools
                if tool.name == tool_name
            )
            result = await selected_tool.ainvoke(tool_args)
            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                )
            )

async def main(argv=None):
    global args
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", help="User prompt about Land Rover Discovery II and its maintenance")
    parser.add_argument("-i", "--interactive", action='store_true', help='Enable interactive chat mode')
    args = parser.parse_args(argv)

    if args.prompt:
        input_prompt = args.prompt
    else:
        input_prompt = "Tell me briefly about land rover discovery 2 model"

    load_dotenv()
    LM_STUDIO_BASEURL = os.getenv("LM_STUDIO_BASEURL")

    model = init_LM_studio_model(
        LM_STUDIO_BASEURL
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["LRDISCO2_MCP_server.py"],
    )


    # Keep MCP connection alive here
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Convert MCP tools -> LangChain tools
            tools = await load_mcp_tools(session)
            print("Available MCP tools:")
            for tool in tools:
                print("-", tool.name)
            if ("search_knowledge_base" in tool.name):
                print("Binding to LR Disco 2 knowledge tool")
                model = model.bind_tools(tools)
            else:
                print("Did not find the Disco2 MCP..is it running?")
                exit()

            if args.interactive:
                while True:
                    input_prompt = input("Ask about Land Rover Discovery 2 (or exit to quit): ")
                    if input_prompt.lower() == 'exit':
                        print("Exiting the interactive prompt. Goodbye!")
                        break
                    reply = await make_query(model, tools, input_prompt)
                    print("\nAnswer:")
                    print(reply)
            else:
                reply = await make_query(model, tools, input_prompt)
                print("\nAnswer:")
                print(reply)

if __name__ == "__main__":
    asyncio.run(main())