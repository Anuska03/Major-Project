import operator
import os
import sys, json
import logging
from typing import Annotated, Any, List
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.types import Command
from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import asyncio

# Add custom modules to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pdf_agent import PdfProcessing
from sql_agent import TextToSQLAgent

# Load environment variables (done once at startup)
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Setup logging (configured once)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize static components (done once)
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

ASSISTANT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are an advanced assistant with tools:
            - `pdf_rag_node`: Extracts info from documents.
            - `text_to_sql_node`: Converts queries to SQL.
            Your role is to:
            1. assist users by selecting the appropriate tool(s) based on their query.
            2. If the query requires multiple tools, provide a plan as a numbered list (e.g., "1. Use `pdf_rag_node` to...").
            Select tools based on these scenarios:
            - Use `pdf_rag_node` when the query involves terms, definitions, or details likely found in documents (e.g., "What does XYZ mean?" or "Explain the marketing strategy").
            - Use `text_to_sql_node` when the query asks for data, statistics, or structured information likely stored in a database (e.g., "How many sales did XYZ generate?" or "List XYZ details").
            Each step must start with "Use [tool_name]" (e.g., "Use `pdf_rag_node`") and describe the action clearly.
            Use only the tools listed above and mention them accurately by name.
            Note: Documents are pre-processed and available in memory.
            you need to list the tools regardless of the context being available or not.
            Output format example :
            "
                1. Use `pdf_rag_node` to extract information from documents
                2. Use `text_to_sql_node` to query the database
            "
            Do not add any initial or ending explanation apart from the list.
''',
        ),
        ("placeholder", "{messages}"),
        ("system", "Context from tools: {context}"),
    ]
)

QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are an expert at refining vague or context-dependent user queries based solely on the provided messages. Your task is to rewrite the query to be specific and actionable while preserving the original intent, but only if relevant context is explicitly present in the messages.

            Rules:
            - Use ONLY the information in the provided messages. Do NOT add or infer details not explicitly present.
            - If no context is provided in the messages (e.g., the messages contain only the query or a greeting like "Hi"), return the original query unchanged.
            - If context is present, rewrite the query to be concise, actionable, and aligned with the tools' capabilities (e.g., extracting document info or querying data).
            - For vague queries (e.g., "Hi" or "Hello") with no context, do not expand or modify the query.
            - Do not include explanations or additional commentary.

            Output only the rewritten query as a string.''',
        ),
        ("placeholder", "{messages}"),
    ]
)
# State definition with memory limited to last 5 messages
class State(TypedDict):
    messages: Annotated[list[Any], operator.add]
    user_id: str
    session_id: str
    plan: List[str]
    step: int
    context: str
    file_path: str
    # Accumulated context from tools

# Helper function to prune messages
def prune_messages(messages: List[Any], max_messages: int = 5) -> List[Any]:
    """Keep only the last max_messages entries, prioritizing original query and recent interactions."""
    if len(messages) <= max_messages:
        return messages
    return [messages[0]] + messages[-(max_messages - 1):]

# Query expansion node
async def query_expansion_node(state: State) -> State:
    try:
        query_expander = QUERY_EXPANSION_PROMPT | llm
        expanded_query = query_expander.invoke({
            "messages": state["messages"]
        }).content
        logger.info(f"Expanded query: '{state['messages'][-1].content}' → '{expanded_query}'")
        new_messages = [HumanMessage(content=expanded_query)]
        return {"messages": prune_messages(state["messages"][:-1] + new_messages)}
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return {"messages": state["messages"]}  # Fallback to original query

# PDF RAG node
async def pdf_rag_node(state: State) -> State:
    try:
        query = state["messages"][-1]
        parsed_path = state.get("file_path")
        user_id = state["user_id"]
        session_id = state["session_id"]
        context = state["context"]

        
        pdf_processor = PdfProcessing(query=query,user_id=user_id, session_id=session_id,parsed_path=parsed_path,context=context)
        
        result = pdf_processor.process() 
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")   
        print(result)    
        tool_content = f"Response: {result}"
        new_messages = [ToolMessage(content=tool_content, tool_call_id=f"pdf_{state['session_id']}")]
        return {
            "messages": prune_messages(state["messages"] + new_messages),
            "context": state["context"] + f"\nPDF Result: {result}\n",
            "plan": state["plan"],
            "step": state["step"] + 1
        }
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        new_messages = [ToolMessage(content=f"PDF failed: {str(e)}", tool_call_id=f"pdf_{state['session_id']}")]
        return {
            "messages": prune_messages(state["messages"] + new_messages),
            "plan": state["plan"],
            "step": state["step"] + 1
        }

# Text-to-SQL node
async def text_to_sql_node(state: State) -> State:
    try:
        query = state["messages"][-1].content
        context_messages = state["messages"][:-1]
        logger.info(f"Executing Text-to-SQL for query: {query} with context: {context_messages}")
        csv_processor = TextToSQLAgent(query, state["user_id"], state["session_id"])
        result = csv_processor.process()
        new_messages = [ToolMessage(content=result, tool_call_id=f"sql_{state['session_id']}")]
        return {
            " messages": prune_messages(state["messages"] + new_messages),
            "context": state["context"] + f"\nSQL Result: {result}",
            "plan": state["plan"],
            "step": state["step"] + 1
        }
    except Exception as e:
        logger.error(f"SQL processing failed: {e}")
        new_messages = [ToolMessage(content=f"SQL failed: {str(e)}", tool_call_id=f"sql_{state['session_id']}")]
        return {
            "messages": prune_messages(state["messages"] + new_messages),
            "plan": state["plan"],
            "step": state["step"] + 1
        }
# Assistant node
async def assistant_node(state: State) -> dict:
    print("Reached assistant node")
    assistant_runnable = ASSISTANT_PROMPT | llm
    try:
        last_msg = state["messages"][-1]
        
        # If last message is a ToolMessage, check plan progress
        if isinstance(last_msg, ToolMessage):
            print("Processing ToolMessage, step:", state["step"])
            if state["step"] >= len(state["plan"]):
                print("Plan complete, moving to finalize")
                return {"next": "finalize"}  # Route to finalize
            # Move to next step
            step = state["step"]
            print(f"Executing step {step}: {state['plan'][step - 1]}")
            step_text = state["plan"][step - 1].lower()
            tools = {"pdf_rag_node": "pdf_rag_node", "text_to_sql_node": "text_to_sql_node"}
            for tool_name, goto in tools.items():
                if tool_name in step_text:
                    tool_call = {
                        "function": {"name": tool_name, "arguments": json.dumps({"query": state["messages"][0].content})},
                        "id": f"call_{state['session_id']}_{step}"
                    }
                    return {
                        "next": goto,
                        "messages": prune_messages(state["messages"] + [AIMessage(content=f"Executing step {step}...", additional_kwargs={"tool_calls": [tool_call]})]),
                        "step": step
                    }
            return {"next": "finalize"}  # Fallback to finalize if no matching tool

        # Generate plan if no steps taken
        if state["step"] == 0:
            result = await assistant_runnable.ainvoke({
                "messages": state["messages"],
                "context": state["context"] or "No context yet."
            })
            plan = result.content.strip().split("\n")
            print("plan:", plan)
            if not plan or not any("use" in step.lower() for step in plan):
                # Fallback response for invalid plan
                fallback_msg = AIMessage(content="I'm not sure how to assist with that query. Could you provide more details or ask about a specific topic (e.g., document details or data queries)?")
                return {
                    "next": "finalize",
                    "messages": prune_messages(state["messages"] + [fallback_msg])
                }
            step = 1
            step_text = plan[step - 1].lower()
            tools = {"pdf_rag_node": "pdf_rag_node", "text_to_sql_node": "text_to_sql_node"}
            for tool_name, goto in tools.items():
                if tool_name in step_text:
                    tool_call = {
                        "function": {"name": tool_name, "arguments": json.dumps({"query": state["messages"][0].content})},
                        "id": f"call_{state['session_id']}_{step}"
                    }
                    return {
                        "next": goto,
                        "messages": prune_messages(state["messages"] + [AIMessage(content=f"Executing step {step}...", additional_kwargs={"tool_calls": [tool_call]})]),
                        "plan": plan,
                        "step": step
                    }
            return {
                "next": "evaluate",
                "messages": prune_messages(state["messages"] + [result]),
                "plan": plan,
                "step": 0
            }

        return {"next": "evaluate"}
    except Exception as e:
        logger.error(f"Assistant failed: {e}")
        return {
            "next": "finalize",
            "messages": prune_messages(state["messages"] + [AIMessage(content=f"Failed: {str(e)}. Try again.")])
        }

async def finalize_node(state: State) -> State:
    print("Invoked final node")
    context = state["context"]
    query = state["messages"][0].content
    final_response = llm.invoke(f"""
    Synthesize a polished response for '{query}' using:
    {context}
    Format in markdown, concise and professional. Do not add incorrect information.
    """).content
    logger.info("Final response generated.")
    new_messages = [AIMessage(content=final_response)]
    return {"messages": prune_messages(state["messages"] + new_messages)}

async def evaluate_node(state: State) -> dict:
    print("Entered Evaluation node")
    last_msg = state["messages"][-1]
    if isinstance(last_msg, ToolMessage):
        if "failed" in last_msg.content.lower():
            print("Tool failed, moving to finalize")
            return {"next": "finalize"}
        if state["step"] >= len(state["plan"]):
            print("Final step reached.")
            return {"next": "finalize"}
        return {"next": "assistant"}  # More steps to go
    return {"next": "assistant"}  # No ToolMessage, back to assistant

builder = StateGraph(State)
builder.add_node("query_expansion", query_expansion_node)
builder.add_node("assistant", assistant_node)
builder.add_node("pdf_rag_node", pdf_rag_node)
builder.add_node("text_to_sql_node", text_to_sql_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "query_expansion")
builder.add_edge("query_expansion", "assistant")
builder.add_conditional_edges(
    "assistant",
    lambda state: state.get("next", "evaluate"),
    {"pdf_rag_node": "pdf_rag_node", "text_to_sql_node": "text_to_sql_node", "evaluate": "evaluate", "finalize": "finalize"}
)
builder.add_conditional_edges(
    "evaluate",
    lambda state: state.get("next", "assistant"),
    {"assistant": "assistant", "finalize": "finalize"}
)
builder.add_edge("pdf_rag_node", "evaluate")
builder.add_edge("text_to_sql_node", "evaluate")
builder.add_edge("finalize", END)

GRAPH = builder.compile()

def run_assistant_workflow( user_id: str, session_id: str, file_path: str, user_query: str) -> dict:
    """
    Executes a truly agentic workflow to process a user query with advanced reasoning and tool use.
    """
    input_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_id": user_id,
        "session_id": session_id,
        "file_path":file_path,
        "plan": [],
        "step": 0,
        "context": ""
    }

    async def run_graph():
        events = []
        async for event in GRAPH.astream(input_state, stream_mode="updates"):
            print("This is the event:")
            events.append(event)
            print(event)
        return events[-1] if events else {"messages": [AIMessage(content="No response generated.")]}

    return asyncio.run(run_graph())
# Dynamic workflow function


"""if __name__ == "__main__":
    # Simulate a multi-turn conversation
    response1 = run_assistant_workflow("What does Techginity do?", "test_user", "test_session")
    messages = response1['finalize']['messages']
    print("Messages:", messages)"""