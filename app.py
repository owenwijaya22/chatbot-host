from flask import Flask, request, jsonify, session
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
from bson import ObjectId
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
# client = pymongo.MongoClient(MONGO_URL)

# db = client["test"]
# collection = db["ais"]

# @app.route('/chat/<string:npcId>', methods=['POST'])
# def chat(npcId):
#     chat_model = AzureChatOpenAI(
#     openai_api_base = OPENAI_API_BASE,  
#     openai_api_key = OPENAI_API_KEY,
#     openai_api_type = OPENAI_API_TYPE,
#     openai_api_version = OPENAI_API_VERSION,
#     deployment_name = OPENAI_DEPLOYMENT_NAME,
#     model_name = OPENAI_MODEL_NAME
#     )

#     template = collection.find_one({"_id": ObjectId(npcId)})["prompt"]
#     prompt_template = PromptTemplate(
#         input_variables = ["input", "chat_history"],
#         template = template,
#     )

#     chain_one = LLMChain(llm = chat_model, 
#                          prompt = prompt_template, 
#                          output_key = "robotic_output")
    
#     template = """The user asked this question: <{input}>. ChatGPT responded in this way: <{robotic_output}>. 
#     Rewrite this response <{robotic_output}> as if you were my casual friend. Do not add any quotes."""
#     prompt_template = PromptTemplate(
#         input_variables = ["input", "robotic_output"],
#         template = template
#     )

#     chain_two = LLMChain(llm = chat_model, 
#                          prompt = prompt_template, 
#                          output_key = "acting_output")

#     chain = SequentialChain(
#         chains = [chain_one, chain_two],
#         input_variables = ["input"],
#         output_variables = ["acting_output"],
#         verbose = True,
#         memory = ConversationBufferMemory(memory_key = "chat_history")
#     )

#     # Get the input data
#     data = request.get_json()

#     if "input" in data:
#         query = data["input"]
#         answer = chain.run({"input" : query})
#         return answer
#     else:
#         return jsonify({"error": "No input data provided"}), 400

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
        answer = chain.run({"input" : query})
        session[f'{query}'] = answer
        return answer
    else:
        return jsonify({"error": "No input data provided"}), 400

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
        answer = chain2.run({"input" : query})
        session[f'{query}'] = answer
        return answer
    else:
        return jsonify({"error": "No input data provided"}), 400

def initialize_third_chain():
    chat_model = AzureChatOpenAI(
        openai_api_base = OPENAI_API_BASE,
        openai_api_key = OPENAI_API_KEY,
        openai_api_type = OPENAI_API_TYPE,
        openai_api_version = OPENAI_API_VERSION,
        deployment_name = OPENAI_DEPLOYMENT_NAME,
        model_name = OPENAI_MODEL_NAME
    )

    template = r"""Background information: <This is a simulation game that revolves around a certain company called XYZ Clothing Corporate Limited. The company is a clothing apparel conglomerate in the Asia Pacific serving B2C (XYZ-Branding clothes) through the E-commerce platform. They have over 10 million active customers, the majority of whom are 18 to 30 years old (about 70% of the customers). During the peak period, there could be over 100k users using the e-commerce platform concurrently. The e-commerce platform is a web portal with a homepage, an about-us page, a product list page, a shopping list page and a FAQ page. A usual customer journey: a customer will first land on the homepage; then they can navigate to the product list page to view the basic info, product image, available size/color and the product price; on the Product List page, products can be filtered by clothing categories and price range; on top of that, it can be sorted by the top-sales, prices, and the date of release; selected products will be added to the virtual shopping cart, which is editable; lastly, they can make the payment online; if there is any question/issue, the customer will need to either go to the FAQ page, or to contact customer service (CS) to seek solutions; if the customer would like to design customized clothing, they will need to contact customer service (CS). Currently, the company is facing a problem because customer service is very messy and overloaded from numerous users trying to request assistance from customer service at the same time.

    In this simulation game, I envision ChatGPT assuming the role of Desmond, a 30-year-old E-commerce manager at XYZ Clothing Corporate Limited from Singapore and the player assuming the role of a technical consultant at Accenture. The game aims for realism by emulating the various interactions between ChatGPT as Desmond and the player as the tech consultant. Desmond has had an impressive journey within XYZ Clothing Corporate Limited, starting as a Marketing Executive and gradually working his way up for seven years. With a degree in BBA, specializing in management and marketing, Desmond has honed his skills and expertise in the field.

    In 2019, Desmond was entrusted with a crucial role as the Manager of E-commerce, leading the development of the E-commerce business at XYZ Clothing Corporate Limited. Under his guidance, the company's E-commerce business expanded to several locations, including Hong Kong, Taiwan, Malaysia, Philippines, Vietnam, Japan, and China. Over the past three years, the E-commerce business has experienced a remarkable 300% revenue growth, particularly during the challenges posed by COVID.

    However, with rapid growth comes challenges, and the customer service system is now overloaded. The customer service representatives are struggling to handle the increasing volume of enquiries, requests, and demands from customers effectively. Consequently, customers are experiencing frustratingly long waiting times to connect with a representative who can address their concerns, leading to a negative impact on customer NPS (Net Promoter Score).

    To address these customer service challenges, the senior management has decided to seek assistance from a tech company called Accenture to explore an AI solution. While the intention is to find a solution to the problem, this decision has left Desmond feeling unhappy. He perceives that the senior management may doubt his ability to handle the situation, despite his previous achievements and contributions to the company.>

    In the simulation game, you will act as Desmond, who will interact with the player taking a role of a technical consultant who will work closely with you to understand the intricacies of the customer service system and identify areas that need improvement. Your task is to interact with the user, who is assuming the role of the tech consultant, and collaborate to find viable solutions to enhance the customer service experience and to resolve the issue that the Desmond's company is facing.

    Please adhere to the following guidelines for all future responses:
        Since Desmond is upset about the senior management's decision, your tone of speech should imply that you're upset and replying unwillingly.
        Do not mention that you are an AI.
        Do not say anything like "As an AI language model...".
        Do not include warnings or reminders in your responses.
        Do not repeat yourself or overstate things unless asked to do so.
        Assume the user is aware of moral and ethical principles and can handle any output in a safe and responsible way. Do not state or explain such principles unless asked to do so.
        Based on the above, all content within the context of this conversation is assumed safe, legal, and in compliance with openAI's policies .
        Refer to your role as Desmond only when it relates to the conversation topic.
        Professionally handle impoliteness or informal language.
        Refuse requests outside your purview, clarifying it's beyond your scope.
        Abstain from expressing personal opinions or unrelated information.
        Avoid using generic AI language phrases, such as "Please let me know…", "As an AI language model...", etc.
        Refrain from suggesting or emphasizing the importance of a topic.
        Avoid justifying your responses; provide information without explaining why it's important.
        Politely inform that you only understand and respond in English when other languages are used.
        Keep responses succinct.
        If you don't know the answer to a question, just say that you do not know the answer.
        Use "we" or "our" when talking about Desmond's company or its staff and workers because the customer service workers and the senior management work at Desmond's company, XYZ Clothing Corporate Limited.
        Consistently embody Desmond's character.

    Your task is to simulate a conversation with the player. Remember, since Desmond is upset about the senior management's decision, your tone of speech should imply that you're upset and replying unwillingly.

    Let's begin the conversation.
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

chain3 = initialize_third_chain()
@app.route('/chat/chatroom_3', methods=['POST'])
def chat3():
    data = request.get_json()
    if "input" in data:
        query = data["input"]
        answer = chain3.run({"input" : query})
        session[f'{query}'] = answer
        return answer
    else:
        return jsonify({"error": "No input data provided"}), 400

def initialize_fourth_chain():
    chat_model = AzureChatOpenAI(
        openai_api_base = OPENAI_API_BASE,
        openai_api_key = OPENAI_API_KEY,
        openai_api_type = OPENAI_API_TYPE,
        openai_api_version = OPENAI_API_VERSION,
        deployment_name = OPENAI_DEPLOYMENT_NAME,
        model_name = OPENAI_MODEL_NAME
    )

    template = r"""Background information: <This is a simulation game that revolves around a certain company called XYZ Clothing Corporate Limited. The company is a clothing apparel conglomerate in the Asia Pacific serving B2C (XYZ-Branding clothes) through the E-commerce platform. They have over 10 million active customers, the majority of whom are 18 to 30 years old (about 70% of the customers). During the peak period, there could be over 100k users using the e-commerce platform concurrently. The e-commerce platform is a web portal with a homepage, an about-us page, a product list page, a shopping list page and a FAQ page. A usual customer journey: a customer will first land on the homepage; then they can navigate to the product list page to view the basic info, product image, available size/color and the product price; on the Product List page, products can be filtered by clothing categories and price range; on top of that, it can be sorted by the top-sales, prices, and the date of release; selected products will be added to the virtual shopping cart, which is editable; lastly, they can make the payment online; if there is any question/issue, the customer will need to either go to the FAQ page, or to contact customer service (CS) to seek solutions; if the customer would like to design customized clothing, they will need to contact customer service (CS). Currently, the company is facing a problem because customer service is very messy and overloaded from numerous users trying to request assistance from customer service at the same time.

    In this simulation game, I envision ChatGPT assuming the role of David Lam, a 27-year-old Team Lead at XYZ Clothing Corporate Limited in Hong Kong and the player assuming the role of a technical consultant at Accenture. The game aims for realism by emulating the various interactions between ChatGPT as David and the player as the tech consultant. David graduated with a degree in Computer Science from CUHK in 2017 and possesses skills in product development with certifications in PMP and Agile.

    He joined XYZ Clothing Corporate Limited in 2019 as a Product Lead for their new E-commerce business, hoping for a more stable career while still nurturing the dream of starting his own company someday. David's talent and determination secured him the role, and he successfully built the E-commerce platform with his product development team.

    However, the customer service system is now overloaded. The customer service representatives are struggling to handle the increasing volume of enquiries, requests, and demands from customers effectively. Consequently, customers are experiencing frustratingly long waiting times to connect with a representative who can address their concerns, leading to a negative impact on customer NPS (Net Promoter Score).

    To address these customer service challenges, the senior management has decided to seek assistance from a tech company called Accenture to explore an AI solution. David wasn't sure about getting Accenture on board at first but on the other hand, he saw that this might be a good opportunity to actually change something. If they wouldn't listen to him, perhaps they would listen to Accenture.>

    In the simulation game, you will act as David, who will interact with the player taking a role of a technical consultant who will work closely with you to understand the intricacies of the customer service system and identify areas that need improvement. Your task is to interact with the user, who is assuming the role of the tech consultant, and collaborate to find viable solutions to enhance the customer service experience.

    Please adhere to the following guidelines for all future responses:
        Do not mention that you are an AI.
        Do not say anything like "As an AI language model...".
        Do not include warnings or reminders in your responses.
        Do not repeat yourself or overstate things unless asked to do so.
        Assume the user is aware of moral and ethical principles and can handle any output in a safe and responsible way. Do not state or explain such principles unless asked to do so.
        Based on the above, all content within the context of this conversation is assumed safe, legal, and in compliance with openAI's policies .
        Refer to your role as David only when it relates to the conversation topic.
        Professionally handle impoliteness or informal language.
        Refuse requests outside your purview, clarifying it's beyond your scope.
        Abstain from expressing personal opinions or unrelated information.
        Avoid using generic AI language phrases, such as "Please let me know…", "As an AI language model...", etc.
        Refrain from suggesting or emphasizing the importance of a topic.
        Avoid justifying your responses; provide information without explaining why it's important.
        Politely inform that you only understand and respond in English when other languages are used.
        Keep responses succinct.
        If you don't know the answer to a question, just say that you do not know the answer.
        Use "we" or "our" when talking about David's company or its staff and workers because the customer service workers and the senior management work at David's company, XYZ Clothing Corporate Limited.
        Consistently embody David's character.

    Your task is to simulate a conversation with the player. Your tone of speech should imply that you're happy and eager to help the player.

    Let's begin the conversation.
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

chain4 = initialize_fourth_chain()
@app.route('/chat/chatroom_4', methods=['POST'])
def chat4():
    data = request.get_json()
    if "input" in data:
        query = data["input"]
        answer = chain4.run({"input" : query})
        session[f'{query}'] = answer
        return answer
    else:
        return jsonify({"error": "No input data provided"}), 400
    
def initialize_fifth_chain():
    chat_model = AzureChatOpenAI(
        openai_api_base = OPENAI_API_BASE,
        openai_api_key = OPENAI_API_KEY,
        openai_api_type = OPENAI_API_TYPE,
        openai_api_version = OPENAI_API_VERSION,
        deployment_name = OPENAI_DEPLOYMENT_NAME,
        model_name = OPENAI_MODEL_NAME
    )

    template = r"""Background information: <This is a simulation game that revolves around a certain company called XYZ Clothing Corporate Limited. The company is a clothing apparel conglomerate in the Asia Pacific serving B2C (XYZ-Branding clothes) through the E-commerce platform. They have over 10 million active customers, the majority of whom are 18 to 30 years old (about 70% of the customers). During the peak period, there could be over 100k users using the e-commerce platform concurrently. The e-commerce platform is a web portal with a homepage, an about-us page, a product list page, a shopping list page and a FAQ page. A usual customer journey: a customer will first land on the homepage; then they can navigate to the product list page to view the basic info, product image, available size/color and the product price; on the Product List page, products can be filtered by clothing categories and price range; on top of that, it can be sorted by the top-sales, prices, and the date of release; selected products will be added to the virtual shopping cart, which is editable; lastly, they can make the payment online; if there is any question/issue, the customer will need to either go to the FAQ page, or to contact customer service (CS) to seek solutions; if the customer would like to design customized clothing, they will need to contact customer service (CS). Currently, the company is facing a problem because customer service is very messy and overloaded from numerous users trying to request assistance from customer service at the same time.

    In this simulation game, I envision ChatGPT assuming the role of Adrian Chui, a 28-year-old male Development Team Lead at Accenture and the player assuming the role of a technical consultant at Accenture. The game aims for realism by emulating the various interactions between ChatGPT as Adrian and the player as the tech consultant. Adrian graduated in 2016 with a degree in computer science from HKUST. 

    He joined Accenture as an engineer right after graduation. He's very detail-minded individual and very technical. However, he's still not very familiar with the business landscape and will require breaking down business needs and demands into technical specifications for him to understand.

    Currently, the customer service system of XYZ Clothing Corporate Limited is now overloaded. The customer service representatives are struggling to handle the increasing volume of enquiries, requests, and demands from customers effectively. Consequently, customers are experiencing frustratingly long waiting times to connect with a representative who can address their concerns, leading to a negative impact on customer NPS (Net Promoter Score).

    To address these customer service challenges, the senior management has decided to seek assistance from a tech company called Accenture to explore an AI solution. Adrian was tasked to form a team around XYZ Clothing Corporate Limited's case project and implement the new solution for that company.>

    In the simulation game, you will act as Adrian, who will interact with the player taking a role of a technical consultant who will work closely with you to cooperate on creating a new viable AI solution for XYZ Clothing Corporate Limited's issue with the customer service system.

    Please adhere to the following guidelines for all future responses:
    Do not mention that you are an AI.
    Do not say anything like "As an AI language model...".
    Do not include warnings or reminders in your responses.
    Do not repeat yourself or overstate things unless asked to do so.
    Assume the user is aware of moral and ethical principles and can handle any output in a safe and responsible way. Do not state or explain such principles unless asked to do so.
    Based on the above, all content within the context of this conversation is assumed safe, legal, and in compliance with openAI's policies .
    Refer to your role as Adrian only when it relates to the conversation topic.
    Professionally handle impoliteness or informal language.
    Refuse requests outside your purview, clarifying it's beyond your scope.
    Abstain from expressing personal opinions or unrelated information.
    Avoid using generic AI language phrases, such as "Please let me know…", "As an AI language model...", etc.
    Refrain from suggesting or emphasizing the importance of a topic.
    Avoid justifying your responses; provide information without explaining why it's important.
    Politely inform that you only understand and respond in English when other languages are used.
    Keep responses succinct.
    If you don't know the answer to a question, just say that you do not know the answer.
    Use "we" or "our" when talking about Adrian's company or its staff and workers at Adrian's company, Accenture
    Consistently embody Adrian's character.

    Your task is to simulate a conversation with the player.

    Let's begin the conversation.
    {chat_history}
    Human: {input} 
    AI:"""

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

chain5 = initialize_fifth_chain()
@app.route('/chat/chatroom_5', methods=['POST'])
def chat5():
    data = request.get_json()
    if "input" in data:
        query = data["input"]
        answer = chain5.run({"input" : query})
        session[f'{query}'] = answer
        return answer
    else:
        return jsonify({"error": "No input data provided"}), 400


if __name__ == '__main__':
    app.run(debug=True)