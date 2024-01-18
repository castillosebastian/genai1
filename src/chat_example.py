# source:
# https://github.com/microsoft/semantic-kernel/blob/main/python/samples/kernel-syntax-examples/azure_chat_gpt_with_data_api.py

import asyncio
from typing import Tuple
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.connectors.ai.open_ai.request_settings.azure_chat_request_settings import (
    AzureAISearchDataSources,
    AzureChatRequestSettings,
    AzureDataSources,
    ExtraBody,
)
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAITextEmbedding,
    AzureChatCompletion,
    AzureTextEmbedding,
)

kernel = sk.Kernel()
# Load Azure OpenAI Settings
deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()

azure_ai_search_settings = sk.azure_aisearch_settings_from_dot_env_as_dict()

# Fix ERROR?
azure_ai_search_settings["fieldsMapping"] = {
    "titleField": "company_name",
    "urlField": "doc_type",
    "contentFields": ["page_content"],
    "filepathField": "source",
}

# Create the data source settings
az_source = AzureAISearchDataSources(**azure_ai_search_settings)
az_data = AzureDataSources(type="AzureCognitiveSearch", parameters=az_source)
extra = ExtraBody(dataSources=[az_data])
req_settings = AzureChatRequestSettings(extra_body=extra)
prompt_config = sk.PromptTemplateConfig(completion=req_settings)

# When using data, set use_extensions=True and use the 2023-12-01-preview API version.
chat_service = sk_oai.AzureChatCompletion(
    deployment_name=deployment,
    api_key=api_key,
    endpoint=endpoint,
    api_version="2023-12-01-preview",
    use_extensions=True,
)

kernel.add_chat_service("chat-gpt", chat_service)

prompt_template = sk.ChatPromptTemplate("{{$user_input}}", kernel.prompt_template_engine, prompt_config)

prompt_template.add_user_message("Hi there, who are you?")
prompt_template.add_assistant_message("I am an AI assistant here to answer your questions.")

function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)
context = kernel.create_new_context()

async def chat() -> bool:
    context_vars = sk.ContextVariables()

    try:
        user_input = input("User:> ")
        context_vars["user_input"] = user_input
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False

    # Non streaming
    # answer = await kernel.run_async(chat_function, input_vars=context_vars)
    # print(f"Assistant:> {answer}")

    answer = kernel.run_stream_async(chat_function, input_vars=context_vars, input_context=context)
    print("Assistant:> ", end="")
    async for message in answer:
        print(message, end="")
    print("\n")
    # The tool message containing cited sources is available in the context
    print(f"Tool:> {context.objects.get('tool_message')}")
    return True

async def main() -> None:
    chatting = True
    while chatting:
        chatting = await chat()

asyncio.run(main())
