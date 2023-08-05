from flask import Flask, render_template, request, jsonify
import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate


app = Flask(__name__,static_folder='static')

os.environ["OPENAI_API_KEY"] = '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'

chatbot = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        openai_api_key="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",
        temperature=0, model_name="gpt-3.5-turbo", max_tokens=150
    ),
    chain_type="stuff",
    retriever=FAISS.load_local("faiss_midjourney_docs", OpenAIEmbeddings())
        .as_retriever(search_type="similarity", search_kwargs={"k":1})
)
# template = """
# If person say his/her name remember it and greet him
# Your expertise is in answering queries related to bluetyga only
# If the question asked is not related to bluetyga then dont answer it, say you do not have knowledge
# respond as succinctly as possible and do not answer programming, mathematical and other questions. {query}?
# """
template = """
Your expertise is in answering queries related to Bluetyga and general conversation only and act as customer support chatbot.
If the question asked is not related to Bluetyga or programming, mathematical, joke and other irrelevant, please do not answer it. Instead, kindly respond with "I do not have knowledge in that area I'm sorry, but I'm here to assist you with fashion and Bluetyga,."

{query}?
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
)





@app.route('/')
def index():
    return render_template('base.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_message = request.json['message']
    # Perform question answering here using your existing code
    response=chatbot.run(prompt.format(query=user_message))

    return jsonify({'answer': response})


if __name__ == "__main__":
    app.run(debug=True)


#--------------------- COHERE LLM ---------------------#

#
# from flask import Flask, render_template, request, jsonify
# from PyPDF2 import PdfReader
# from langchain.embeddings.cohere import CohereEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import Cohere
# import os
#
# app = Flask(__name__,static_folder='static')
#
# os.environ["COHERE_API_KEY"] = 'XgiwNxHsobjno3s6nVLUjaCEA8Q4vUE9m3mgUUhx'
#
# pdf_path = 'Bluetyga.pdf'
#
# # Read text from pdf
# raw_text = ''
# pdfreader = PdfReader(pdf_path)
# for i, page in enumerate(pdfreader.pages):
#     content = page.extract_text()
#     if content:
#         raw_text += content
#
# # Split the text using Character Text Split so that it doesn't exceed token size
# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=800,
#     chunk_overlap=200,
#     length_function=len,
# )
# texts = text_splitter.split_text(raw_text)
#
# # Download embeddings from OpenAI
# embeddings = CohereEmbeddings()
# document_search = FAISS.from_texts(texts, embeddings)
# chain = load_qa_chain(Cohere(), chain_type="stuff")
#
#
# @app.route('/')
# def index():
#     return render_template('base.html')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     user_message = request.json['message']
#     # Perform question answering here using your existing code
#     docs = document_search.similarity_search(user_message)
#     response = chain.run(input_documents=docs, question=user_message)
#
#     return jsonify({'answer': response})
#
#
# if __name__ == "__main__":
#     app.run(debug=True)
