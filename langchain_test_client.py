#Simple local MCP server test
import argparse
import os
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from LRDISCO2_MCP_server import search_knowledge_base

def init_LM_studio_model(base_url):
    model = init_chat_model(
        model="qwen/qwen3-vl-8b",
        model_provider="openai",
        base_url=base_url,
        api_key="not-needed"
    )
    return model

def make_query(model, user_query, mcp_result):
    messages = [
        SystemMessage(
            content=(
                "You are a helpful car mechanics assistant. "
                "Stay concise and to the point."
            )
        ),
        HumanMessage(
            content=f"""
            User question:
            {user_query}
            Background information retrieved from the local MCP server:
            {mcp_result}
            Use the background information when relevant. If it does not contain
            the answer, use your general knowledge.
            """
        )
    ]

    response = model.invoke(messages)
    return response.content

def main(argv=None):
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
    LM_STUDIO_BASEURL = os.getenv('LM_STUDIO_BASEURL') #http://<Check your local LM Studio setup for IP>:1234/v1

    model = init_LM_studio_model(LM_STUDIO_BASEURL)

    if args.interactive:
        while True:
            input_prompt = input("Ask about Land Rover Discovery 2 (or exit to quit): ")
            if input_prompt.lower() == 'exit':
                print("Exiting the interactive prompt. Goodbye!")
                break
            knowledge = search_knowledge_base(input_prompt)
            reply = make_query(model, input_prompt, knowledge)
            print(reply)
    else:
        knowledge = search_knowledge_base(input_prompt)
        reply = make_query(model, input_prompt, knowledge)
        print(reply)


if __name__ == "__main__":
    main()