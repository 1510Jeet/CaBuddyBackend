from fastapi import FastAPI,HTTPException
from pydantic import BaseModel

import os



#to avoid this error: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#for LLM 
os.environ["GROQ_API_KEY"] = "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"
 
from langchain_groq import ChatGroq
 
llmModel = ChatGroq(model="llama3-8b-8192")
 
from langchain_core.messages import HumanMessage
 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
 
store = {}




#initialize the app
app = FastAPI()





# Now on to making a llama3 API

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(llmModel, get_session_history)



#Request body of caBuddy enpoint message ie the user prompt and session_id of a particular session
class Message(BaseModel):
    message: str
    session_id:str

@app.post("/caBuddy/")
async def llmResponse(msg: Message):
    message=msg.message
    session_id=msg.session_id
    prompt = "Your name is 'CA AI Buddy for Tally Solutions'. Only answer questions directly related to Chartered Accountancy know-how only. Refrain from answering any other questions about people, companies, etc. kindly."
    prompt+= "Refrain from answering questions about Tally or any other companies or topics unrelated to Chartered Accounting knowledge and know-how."
    # prompt+= "Respond very very kindly to derogatory remarks and foul language and ask them where you can be helpful in a short response."
    prompt+="Respond in a very kind and short one sentence way to derogatory remarks and ask them where you can help them."
    message = prompt + "\n" + message
    response = with_message_history.invoke(
        [HumanMessage(content=message)],
        config = {"configurable": {"session_id": session_id}},
    )
    return response.content


#declaring the session_id parameter required by deleteChat API enpoint
class sessionInfo(BaseModel):
    session_id:str

@app.post("/deleteChat")
async def deleteChat(sid:sessionInfo):
    # Deleting Chat by it's session id
    session_id=sid.session_id
    if session_id in store:
        del store[session_id]





if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="127.0.0.1", port=8000)
