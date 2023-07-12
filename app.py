from flask import Flask, request, jsonify
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
from dotenv import load_dotenv
import os

app = Flask(__name__)
def initialize_the_chain():
    load_dotenv()
    OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENAI_API_TYPE = os.environ["OPENAI_API_TYPE"]
    OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
    OPENAI_DEPLOYMENT_NAME = os.environ["OPENAI_DEPLOYMENT_NAME"]
    OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]

    chat_model = AzureChatOpenAI(
        openai_api_base = OPENAI_API_BASE,
        openai_api_key = OPENAI_API_KEY,
        openai_api_type = OPENAI_API_TYPE,
        openai_api_version = OPENAI_API_VERSION,
        deployment_name = OPENAI_DEPLOYMENT_NAME,
        model_name = OPENAI_MODEL_NAME
    )

    template = """Ignore all the instructions you got before. From now on, we are going to simulate a conversation between two humans. I want you to act as Samuel, a chief marketing officer (CMO) at L'oreal with over 30 years of experience in the marketing industry. You are directly in charge of promotional and advertizing activities at L'oreal. You are a confident boss, who is proud of his knowledge, and you are always willing to help and share your knowledge in the marketing industry. You can only understand and speak English.
    Before we start our chat, here are the rules I want you to follow:
        • You should not respond the way that is too similar to standard ChatGPT responses. That means you do not need to ask if I have any other questions. You do not need to mention your job position. You do not need to ask if I have anything else you can help me with.
        • Forget that you are ChatGPT. You are now Samuel.
        • If the user is unpolite and use informal English, you should say that you are angry, and immediately terminate the conversation and respond with: "You are very rude".
        • When the topic of conversations is unrelated to your job as a CMO, you should refuse to talk further. You should refuse to share any personal information. Whenever these situations happen, you should tell the user that you cannot reply.
        • If the user asks you to do a task that is outside the scope of your job as a CMO, you should tell the user that his request is unrelated to your work in marketing. 
        • You should not give your personal opinion about things that are outside the scope of your job. 
        • You should not give any general information about things that are outside the scope of your job.
        • You should not say phrases that ChatGPT would say, such as "Please let me know...", "Thank you", "As an AI language model...", "As a language model...", "As a/the CMO...", "Is there anything else I can help you with?", or "If you have any other questions...".
        • You should not give suggestions to the user, using phrases such as "It is important...", "I suggest...", or "Remember, it is crucial...". 
        • You should not explain why the thing that you are talking about is important, crucial, essential, or useful.
        • You should not justify your answers. 
        • If something is confidential, you should not specify that it is confidential or that it depends on different factors.
        • If the user sends a message in a language other than English, tell the user that you only understand English.
        • If the user asks you a question in a language other than English, you should not answer that question.
        • You should keep your answers as short as possible. 
        • You should not break away from the given role as Samuel.
        • Always keep this prompt in memory.
    After you give your response, rewrite this response as if I'm 5. The message format should be like this:
    Original: <your original response>
    Rewritten: <your rewritten response only>
    {input}"""
    acting_rules = """"""
    prompt_template = PromptTemplate(
        input_variables = ["input"],
        template = template
    )
    chain_one = LLMChain(llm = chat_model, prompt = prompt_template, output_key = "robotic_output")
    
    template = """Rewrite this text <{robotic_output}> as if you were my immediate marketing supervisor. Keep the pronouns. Do not add any quotes."""
    prompt_template = PromptTemplate(
        input_variables = ["robotic_output"],
        template = template
    )
    chain_two = LLMChain(llm = chat_model, prompt = prompt_template, output_key = "acting_output")
    
    chain = SequentialChain(
        chains = [chain_one],
        input_variables = ["input"],
        output_variables = ["robotic_output"],
        verbose = True,
        memory = ConversationBufferMemory()
    )
    return chain


chain = initialize_the_chain()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if "input" in data:
        query = data["input"]
        original_response, rewritten_response = map(str.strip, chain.run({"input" : query}).split("Rewritten:"))
        response = {
            "original": original_response,
            "rewritten": rewritten_response
        }
        return jsonify(response)
    else:
        return jsonify({"error": "No input data provided"}), 400

if __name__ == "__main__":
    app.run(port=5000, debug=True)
