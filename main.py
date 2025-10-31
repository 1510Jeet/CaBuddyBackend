from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import certifi
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from typing import List

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# To avoid this error: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Environment Checks ---
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    raise ValueError("MONGODB_URI not found in environment variables")

# --- LLM and LangChain Setup ---
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# --- MongoDB Configuration ---
DATABASE_NAME = "ca_assistant_db"
COLLECTION_NAME = "chat_sessions"

# --- Constants for Context Management ---
MAX_CONTEXT_TOKENS = 6000  # Conservative limit for the model
MAX_MESSAGES_IN_CONTEXT = 30  # Maximum messages to keep in active context
SUMMARY_TRIGGER_MESSAGES = 20  # When to trigger summarization
KEEP_RECENT_MESSAGES = 8  # Recent messages to always keep unsummarized

# --- Token Estimation (approximate) ---
def estimate_tokens(text: str) -> int:
    """Rough estimation: 1 token ≈ 4 characters"""
    return len(text) // 4

def estimate_messages_tokens(messages: List[BaseMessage]) -> int:
    """Estimate total tokens in message list"""
    return sum(estimate_tokens(str(m.content)) for m in messages)

# --- Improved History Management with Smart Context Window ---
def get_session_history(session_id: str):
    """Get session history with intelligent context management"""
    history = MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=mongodb_uri,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME
    )
    return history

def prepare_context_messages(session_id: str) -> List[BaseMessage]:
    """
    Prepare optimized context messages with smart truncation and summarization.
    This ensures we stay within token limits while preserving important context.
    """
    history = get_session_history(session_id)
    messages = history.messages
    
    if not messages:
        return []
    
    # If we have few messages, return all
    if len(messages) <= KEEP_RECENT_MESSAGES:
        return messages
    
    # Check if we need to summarize
    if len(messages) > SUMMARY_TRIGGER_MESSAGES:
        logging.info(f"Session {session_id}: Triggering summarization ({len(messages)} messages)")
        
        try:
            # Keep the most recent messages
            recent_messages = messages[-KEEP_RECENT_MESSAGES:]
            older_messages = messages[:-KEEP_RECENT_MESSAGES]
            
            # Create summary of older messages
            summary = create_conversation_summary(older_messages, session_id)
            
            if summary:
                # Create summary message
                summary_msg = SystemMessage(
                    content=f"[Previous conversation context]: {summary}"
                )
                # Return summary + recent messages
                optimized_messages = [summary_msg] + recent_messages
                
                # Verify token count
                total_tokens = estimate_messages_tokens(optimized_messages)
                logging.info(f"Session {session_id}: Optimized to ~{total_tokens} tokens")
                
                return optimized_messages
            else:
                # If summarization fails, fall back to recent messages only
                logging.warning(f"Session {session_id}: Summarization failed, using recent messages only")
                return recent_messages
                
        except Exception as e:
            logging.error(f"Error in prepare_context_messages for {session_id}: {e}")
            # Fallback to recent messages
            return messages[-KEEP_RECENT_MESSAGES:]
    
    # If not enough messages to summarize, use sliding window
    if len(messages) > MAX_MESSAGES_IN_CONTEXT:
        logging.info(f"Session {session_id}: Truncating to last {MAX_MESSAGES_IN_CONTEXT} messages")
        return messages[-MAX_MESSAGES_IN_CONTEXT:]
    
    return messages

def create_conversation_summary(messages: List[BaseMessage], session_id: str) -> str:
    """Create a concise summary of conversation messages"""
    try:
        # Use lightweight model for summarization
        summary_llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.3,
            max_tokens=200
        )
        
        # Build conversation string
        convo_parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                convo_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                convo_parts.append(f"Assistant: {msg.content}")
        
        convo_text = "\n".join(convo_parts)
        
        # Create summary prompt
        summary_prompt = f"""Summarize this CA/Tally conversation into 3-4 key points:

{convo_text}

Provide a concise summary focusing on:
- Main topics discussed
- Key questions asked
- Important advice given
- Any specific Tally or CA procedures mentioned

Summary:"""
        
        response = summary_llm.invoke([HumanMessage(content=summary_prompt)])
        summary = response.content.strip()
        
        logging.info(f"Created summary for session {session_id}: {len(summary)} chars")
        return summary
        
    except Exception as e:
        logging.error(f"Summarization error for session {session_id}: {e}")
        return None

# --- System Prompt ---
SYSTEM_PROMPT_CONTENT = """You are 'AI Buddy for Accounting Queries' - an expert assistant for Accounting.

Your role:
- Answer questions about CA practices, accounting principles, taxation, and IT means Income Tax.
- Provide accurate, professional guidance on accounting and finance topics


Guidelines:
- Only answer questions related to Chartered Accountancy, accounting and taxation.
- For off-topic questions: politely redirect to CA, Accounting topics in one friendly sentence
- For inappropriate remarks: respond kindly in one short sentence and ask how you can help
- Be concise but thorough - provide actionable advice
- If you're unsure, acknowledge it and suggest consulting official resources
- Always give answers less than 300 words unless asked to be more detailed
- Answer in 4-5 Lines unless asked.
- Always give short answers unless asked to give detailed answers.
- Even when detailed answers are asked don't give answers longer than 300 words unless it's asked to explain it in very detailed.
Stay focused on your expertise: CA knowledge and Accounting."""

# --- LLM Initialization with Improved Chain ---
try:
    # Main LLM for conversations
    llm_model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        max_tokens=1000  # Reasonable response length
    )
    
    # Create prompt template with system prompt and history placeholder
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_CONTENT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Chain: Prompt -> LLM -> Parser
    chain = prompt_template | llm_model | StrOutputParser()
    
    logging.info("LLM chain initialized successfully")
    
except Exception as e:
    logging.error(f"Failed to initialize LLM chain: {e}")
    raise

# --- Custom Runnable with Context Management ---
def get_managed_history(session_id: str):
    """Wrapper that returns managed history"""
    # This returns the actual history object for saving new messages
    return get_session_history(session_id)

# Create the runnable with message history
with_message_history = RunnableWithMessageHistory(
    chain,
    get_managed_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- FastAPI App Initialization ---
app = FastAPI(title="CA AI Buddy Backend", version="3.0")

# --- Startup Event for DB Connection Test ---
@app.on_event("startup")
async def startup_db_client():
    try:
        ca_bundle = certifi.where()
        client = MongoClient(mongodb_uri, tlsCAFile=ca_bundle, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db_names = client.list_database_names()
        logging.info("✅ Successfully connected to MongoDB.")
        logging.info(f"Available databases: {db_names}")
        client.close()
    except ConnectionFailure as e:
        logging.error(f"❌ MongoDB connection failed: {e}")
    except Exception as e:
        logging.error(f"❌ Unexpected error during MongoDB connection test: {e}")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Models ---
class Message(BaseModel):
    message: str
    session_id: str

class SessionInfo(BaseModel):
    session_id: str

# --- API Endpoints ---
@app.post("/caBuddy/")
async def llm_response(msg: Message):
    """Main endpoint for CA AI Buddy conversations with improved context management"""
    message_content = msg.message.strip()
    if not message_content:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    session_id = msg.session_id
    logging.info(f"📨 Received message for session: {session_id}")
    
    try:
        # Prepare optimized context
        context_messages = prepare_context_messages(session_id)
        
        # Log context info
        context_tokens = estimate_messages_tokens(context_messages)
        logging.info(f"Context: {len(context_messages)} messages, ~{context_tokens} tokens")
        
        # Invoke with optimized history
        response = with_message_history.invoke(
            {
                "input": message_content,
                "history": context_messages  # Pass pre-processed context
            },
            config={"configurable": {"session_id": session_id}},
        )
        
        logging.info(f"✅ Successfully generated response for session: {session_id}")
        return response
        
    except Exception as e:
        logging.error(f"❌ LLM invocation error for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

@app.post("/deleteChat")
async def delete_chat(sid: SessionInfo):
    """Delete chat history for a session"""
    session_id = sid.session_id
    logging.info(f"🗑️ Attempting to delete chat history for session: {session_id}")
    
    try:
        history = get_session_history(session_id)
        message_count = len(history.messages)
        history.clear()
        logging.info(f"✅ Successfully cleared {message_count} messages for session: {session_id}")
        return {"status": "success", "messages_deleted": message_count}
    except Exception as e:
        logging.error(f"❌ Failed to delete chat for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delete Error: {str(e)}")

@app.get("/chatStats/{session_id}")
async def get_chat_stats(session_id: str):
    """Get statistics about the chat session"""
    try:
        history = get_session_history(session_id)
        messages = history.messages
        
        total_messages = len(messages)
        total_tokens = estimate_messages_tokens(messages)
        
        user_messages = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_messages = sum(1 for m in messages if isinstance(m, AIMessage))
        
        return {
            "session_id": session_id,
            "total_messages": total_messages,
            "user_messages": user_messages,
            "ai_messages": ai_messages,
            "estimated_tokens": total_tokens,
            "status": "active" if total_messages > 0 else "empty"
        }
    except Exception as e:
        logging.error(f"❌ Failed to get stats for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Stats Error: {str(e)}")

# --- Test Endpoint for DB Writing ---
@app.post("/test-db-write")
async def test_db_write(sid: SessionInfo):
    """Test endpoint to verify MongoDB write functionality"""
    session_id = sid.session_id
    logging.info(f"🧪 Running DB write test for session: {session_id}")
    
    try:
        history = get_session_history(session_id)
        test_messages = [
            HumanMessage(content="Test message from user - DB write verification"),
            AIMessage(content="Test response from AI - DB write verification")
        ]
        history.add_messages(test_messages)
        
        # Verify by loading back
        loaded = history.messages
        logging.info(f"✅ DB write test successful for session: {session_id} ({len(loaded)} messages)")
        
        return {
            "status": "success",
            "detail": "Test messages written and verified",
            "loaded_count": len(loaded)
        }
    except Exception as e:
        logging.error(f"❌ DB write test failed for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB Write Test Error: {str(e)}")

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Quick MongoDB ping
        ca_bundle = certifi.where()
        client = MongoClient(mongodb_uri, tlsCAFile=ca_bundle, serverSelectionTimeoutMS=3000)
        client.admin.command('ping')
        client.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "llm": "ready"
        }
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# --- Main Execution ---
if __name__ == "__main__":
    # Test MongoDB connection
    ca_bundle = certifi.where()
    try:
        client = MongoClient(mongodb_uri, tlsCAFile=ca_bundle)
        db_names = client.list_database_names()
        print("\n✅ Successfully connected to MongoDB Atlas.")
        print("Available databases:")
        for name in db_names:
            print(f"  - {name}")
        client.close()
    except ConnectionFailure as e:
        print("\n❌ MongoDB connection failed.")
        print(f"Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

    # Run the app
    # import uvicorn
    # uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")