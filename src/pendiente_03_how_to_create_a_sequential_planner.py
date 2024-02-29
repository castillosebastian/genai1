import asyncio
import os 
import sys
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from semantic_kernel.core_skills import TextSkill
from semantic_kernel.planning import SequentialPlanner, Plan
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

with open("./semantic_kernel_examples/examples/prompts/sk_seq_prompt", "r") as f:
    PROMPT = f.read()

kernel = sk.Kernel()

api_key, org_id = sk.azure_aisearch_settings_from_dot_env()
deployment_name, key, endpoint = sk.azure_openai_settings_from_dot_env()
embeddings = os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"]

azure_chat_service = AzureChatCompletion(deployment_name=deployment_name, endpoint=endpoint, api_key=key)
azure_text_embedding = AzureTextEmbedding(deployment_name=embeddings, endpoint=endpoint, api_key=key)

kernel.add_chat_service(
        "chat_completion",
        AzureChatCompletion(deployment_name=deployment_name, endpoint=endpoint, api_key=key),
)

skills_directory = "./semantic_kernel_examples/skills/"
writer_skill = kernel.import_semantic_skill_from_directory(skills_directory, "WriterSkill")
# writer_skill = kernel.import_semantic_skill_from_directory(skills_directory, "WriterSkill")
summarize_skill = kernel.import_semantic_skill_from_directory(skills_directory, "SummarizeSkill")
text_skill = kernel.import_skill(TextSkill(), "TextSkill")
# sk_prompt = """
# {{$input}}
#
# Rewrite the above in the style of Shakespeare.
# """
# shakespeareFunction = kernel.create_semantic_function(sk_prompt, "shakespeare", "ShakespeareSkill",
#                                                       max_tokens=2000, temperature=0.8)
ask = """
Tomorrow is Valentine's day. I need to come up with a few date ideas.
Convert the text to lowercase, summarize the text and then convert to french."""


planner = SequentialPlanner(kernel,prompt=PROMPT)

sequential_plan = asyncio.run(planner.create_plan_async(goal=ask))

# for step in sequential_plan._steps:
#     print(step.description, ":", step._state.__dict__)
#
result = asyncio.run(sequential_plan.invoke_async())
print("final result is ", result)
#
# print(result)