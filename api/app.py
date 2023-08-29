from flask import Flask, request, jsonify
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
from bson.objectid import ObjectId
import os
import pymongo
import dotenv
from flask_cors import CORS

dotenv.load_dotenv()


MONGO_URL = os.environ["MONGO_URL"]
OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_TYPE = os.environ["OPENAI_API_TYPE"]
OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
OPENAI_DEPLOYMENT_NAME = os.environ["OPENAI_DEPLOYMENT_NAME"]
OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]

app = Flask(__name__)
CORS(app)

chains = {}
client = pymongo.MongoClient(MONGO_URL)

db = client["test"]
collection = db["ais"]

@app.route('/chat/<string:npcId>', methods=['POST'])
def chats(npcId):
    chat_model = AzureChatOpenAI(
    openai_api_base = OPENAI_API_BASE,  
    openai_api_key = OPENAI_API_KEY,
    openai_api_type = OPENAI_API_TYPE,
    openai_api_version = OPENAI_API_VERSION,
    deployment_name = OPENAI_DEPLOYMENT_NAME,
    model_name = OPENAI_MODEL_NAME
    )

    template = collection.find_one({"_id": ObjectId(npcId)})["prompt"]
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

    # Get the input data
    data = request.get_json()

    if "input" in data:
        query = data["input"]
        answer = chain.run({"input" : query})
        return answer
    else:
        return jsonify({"error": "No input data provided"}), 400
