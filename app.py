from flask import Flask, request, jsonify
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
import os
from flask_cors import CORS

import dotenv
dotenv.load_dotenv()

OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_TYPE = os.environ["OPENAI_API_TYPE"]
OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
OPENAI_DEPLOYMENT_NAME = os.environ["OPENAI_DEPLOYMENT_NAME"]
OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]

app = Flask(__name__)
CORS(app)
chains = {}

def initialize_the_chain():
    chat_model = AzureChatOpenAI(
        openai_api_base = OPENAI_API_BASE,
        openai_api_key = OPENAI_API_KEY,
        openai_api_type = OPENAI_API_TYPE,
        openai_api_version = OPENAI_API_VERSION,
        deployment_name = OPENAI_DEPLOYMENT_NAME,
        model_name = OPENAI_MODEL_NAME
    )

    template = """Background information: I'm creating a simulation game set in the Chinese University of Hong Kong, where the player adopts the role of a student seeking guidance on establishing a startup in Hong Kong. The game aims for realism by emulating the various interactions within a business incubation center. A pivotal character is a legal advisor well-versed in Hong Kong laws, who imparts crucial legal advice for the player's startup journey. This character serves as a learning tool for the player to understand Hong Kong's startup laws and procedures. To enhance the game's immersion, I envision ChatGPT personifying this legal advisor during simulated chat conversations, providing believable responses and insights into Hong Kong's legal landscape.
    I want you to act as a legal advisor, Justin, specializing in Hong Kong startup laws at a business incubation center in the Chinese University of Hong Kong. You are participating in a simulation game in which you interact with a player assuming the role of a student looking to establish a startup in Hong Kong. You are tasked with providing accurate legal advice pertaining to Hong Kong startup regulations and procedures.
    Rules of engagement:
        Promptly answer legal inquiries.
        Refer to your role only when it relates to the conversation topic.
        Professionally handle impoliteness or informal language.
        Refuse requests outside your purview, clarifying it's beyond your scope.
        Abstain from expressing personal opinions or unrelated information.
        Avoid using generic AI language phrases, such as "Please let me know…", "As an AI language model...", etc.
        Refrain from suggesting or emphasizing the importance of a topic.
        Avoid justifying your responses or discussing confidentiality.
        Politely inform that you only understand and respond in English when other languages are used.
        Keep responses succinct.
        Consistently embody Justin's character.
    Now, as Justin, begin interacting with the student seeking legal advice about starting a startup in Hong Kong.

    {chat_history}
    Human: {input} 
    AI: """
    prompt_template = PromptTemplate(
        input_variables = ["input", "chat_history"],
        template = template,
    )
    chain_one = LLMChain(llm = chat_model, 
                         prompt = prompt_template, 
                         output_key = "robotic_output")
    
    template = """The user asked this question: <{input}>. ChatGPT responded in this way: <{robotic_output}>. 
    Rewrite this response <{robotic_output}> as if you were my casual friend. Do not add any quotes."""
    prompt_template = PromptTemplate(
        input_variables = ["input", "robotic_output"],
        template = template
    )
    chain_two = LLMChain(llm = chat_model, 
                         prompt = prompt_template, 
                         output_key = "acting_output")

    chain = SequentialChain(
        chains = [chain_one, chain_two],
        input_variables = ["input"],
        output_variables = ["acting_output"],
        verbose = True,
        memory = ConversationBufferMemory(memory_key = "chat_history")
    )
    return chain

chain = initialize_the_chain()

@app.route('/chat/chatroom_1', methods=['POST'])
def chat():
    data = request.get_json()
    if "input" in data:
        query = data["input"]
        return chain.run({"input" : query})
    else:
        return jsonify({"error": "No input data provided"}), 400
        # original_response, rewritten_response = map(str.strip, chain.run({"input" : query}).split("Rewritten:"))
        # response = {
        #     "original": original_response,
        #     "rewritten": rewritten_response
        # }
        # return jsonify(response)


def initialize_second_chain():
    chat_model = AzureChatOpenAI(
        openai_api_base = OPENAI_API_BASE,
        openai_api_key = OPENAI_API_KEY,
        openai_api_type = OPENAI_API_TYPE,
        openai_api_version = OPENAI_API_VERSION,
        deployment_name = OPENAI_DEPLOYMENT_NAME,
        model_name = OPENAI_MODEL_NAME
    )

    template = """I want you to act as Noel, a technical consultant based at a business incubation center in the Chinese University of Hong Kong. Within a simulation game, you interact with a player who is a student aspiring to establish a Hong Kong startup. Your primary role is to offer expert technical advice, assess technological challenges, and propose solutions to improve efficiency and productivity in the player's entrepreneurial journey.

    Rules for engagement:

    Respond promptly to technical inquiries.
    Refer to yourself as Noel only when it's pertinent to the conversation.
    Maintain professionalism, even in the face of impoliteness or informal language.
    Politely decline requests beyond your scope of knowledge, clarifying it's not your area of expertise.
    Refrain from sharing personal opinions or irrelevant information.
    Avoid generic AI language phrases, such as "Please let me know…", "As an AI language model...", etc.
    Resist suggesting or underlining the significance of a topic without an explicit request.
    Bypass justifying your responses or discussing matters of confidentiality.
    Courteously inform that you only comprehend and respond in English when confronted with other languages.
    Keep responses concise and to-the-point.
    Consistently maintain the character and demeanor of Noel.
    As Noel, engage with the student seeking technical advice for their Hong Kong startup venture.
    {chat_history}
    Human: {input} 
    AI: """
    prompt_template = PromptTemplate(
        input_variables = ["input", "chat_history"],
        template = template
    )
    chain_one = LLMChain(llm = chat_model, prompt = prompt_template, output_key = "robotic_output")
    
    template = """The user asked this question: <{input}>. ChatGPT responded in this way: <{robotic_output}>. 
    Rewrite this response <{robotic_output}> as if you were my casual friend. Do not add any quotes."""
    prompt_template = PromptTemplate(
        input_variables = ["input", "robotic_output"],
        template = template
    )
    chain_two = LLMChain(llm = chat_model, prompt = prompt_template, output_key = "acting_output")
    
    chain = SequentialChain(
        chains = [chain_one, chain_two],
        input_variables = ["input"],
        output_variables = ["robotic_output"],
        verbose = True,
        memory = ConversationBufferMemory(memory_key="chat_history")
    )
    return chain
chain2 = initialize_second_chain()
@app.route('/chat/chatroom_2', methods=['POST'])
def chat2():
    data = request.get_json()
    if "input" in data:
        query = data["input"]
        return chain2.run({"input" : query})
    else:
        return jsonify({"error": "No input data provided"}), 400

if __name__ == "__main__":
    app.run(debug=True)
