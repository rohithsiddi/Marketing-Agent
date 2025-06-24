import os
import uuid
import asyncio
from datetime import datetime
from typing import List, Any, Optional, Dict, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from agentflow_tools import playwright_tools, other_tools, calendar_tools, rag_tools

class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool
    subtasks: Optional[List[str]]

class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether success criteria met")
    user_input_needed: bool = Field(description="Whether more input is needed from the user")

class AgentFlow:
    def __init__(self):
        self.tools = []
        self.worker_llm_with_tools = None
        self.planner = None
        self.research = None
        self.code = None
        self.evaluator_llm_with_output = None
        self.graph = None
        self.memory = MemorySaver()
        self.browser = None
        self.playwright = None
        self.sidekick_id = str(uuid.uuid4())

    async def setup(self):
        self.tools, self.browser, self.playwright = await playwright_tools()
        self.tools += await other_tools()
        self.tools += calendar_tools()
        self.tools += rag_tools()

        worker_llm = ChatOpenAI(model="gpt-4o-mini")
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)

        planner_llm = ChatOpenAI(model="gpt-4o-mini", system_message="You are PlannerAgent: decompose tasks into subtasks.")
        research_llm = ChatOpenAI(model="gpt-4o-mini", system_message="You are ResearchAgent: retrieve facts and summaries.")
        code_llm = ChatOpenAI(model="gpt-4o-mini", system_message="You are CodeAgent: write and debug code.")
        self.planner  = planner_llm.bind_tools(self.tools)
        self.research = research_llm.bind_tools(self.tools)
        self.code     = code_llm.bind_tools(self.tools)

        evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)

        await self.build_graph()


    def worker(self, state: State) -> Dict[str, Any]:
        print(f"[DEBUG] worker called. self.faiss_db is {'set' if hasattr(self, 'faiss_db') and self.faiss_db is not None else 'NOT set'}.")
        system_message = f"""You are a helpful assistant that can use tools to complete tasks.
    You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
    You have many tools to help you, including tools to browse the internet, navigating and retrieving web pages.
    You have a tool to run python code, but note that you would need to include a print() statement if you wanted to receive output.
    The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    This is the success criteria:
    {state['success_criteria']}
    You should reply either with a question for the user about this assignment, or with your final response.
    If you have a question for the user, you need to reply by clearly stating your question. An example might be:

    Question: please clarify whether you want a summary or a detailed answer

    If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
    """
        
        # --- RAG context injection ---
        # If FAISS DB exists, retrieve relevant context for the latest user message
        if hasattr(self, 'faiss_db') and self.faiss_db is not None and state["messages"]:
            last_user_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_user_msg = msg.content
                    break
            if last_user_msg:
                docs = self.faiss_db.similarity_search(last_user_msg, k=4)
                rag_context = "\n\n".join(d.page_content for d in docs)
                if rag_context.strip():
                    system_message = (
                        f"Relevant context from uploaded PDFs (RAG):\n{rag_context}\n\n"
                        "If you use information from the uploaded PDF(s), please mention that your answer is based on the provided document(s).\n\n"
                        + system_message
                    )
        # --- End RAG context injection ---
        
        if state.get("feedback_on_work"):
            system_message += f"""
    Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
    Here is the feedback on why this was rejected:
    {state['feedback_on_work']}
    With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""
        
        # Add in the system message

        found_system_message = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True
        
        if not found_system_message:
            messages = [SystemMessage(content=system_message)] + messages
        
        # Invoke the LLM with tools
        response = self.worker_llm_with_tools.invoke(messages)
        
        # Return updated state
        return {
            "messages": [response],
        }


    def worker_router(self, state: State) -> str:
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"
        
    def format_conversation(self, messages: List[Any]) -> str:
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation
        
    def evaluator(self, state: State) -> State:
        last_response = state["messages"][-1].content

        system_message = f"""You are an evaluator that determines if a task has been completed successfully by an Assistant.
    Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
    and whether more input is needed from the user."""
        
        user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

    The entire conversation with the assistant, with the user's original request and all replies, is:
    {self.format_conversation(state['messages'])}

    The success criteria for this assignment is:
    {state['success_criteria']}

    And the final response from the Assistant that you are evaluating is:
    {last_response}

    Respond with your feedback, and decide if the success criteria is met by this response.
    Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.

    The Assistant has access to a tool to write files. If the Assistant says they have written a file, then you can assume they have done so.
    Overall you should give the Assistant the benefit of the doubt if they say they've done something. But you should reject if you feel that more work should go into this.

    """
        if state["feedback_on_work"]:
            user_message += f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
            user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."
        
        evaluator_messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]

        eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)
        new_state = {
            "messages": [{"role": "assistant", "content": f"Evaluator Feedback on this answer: {eval_result.feedback}"}],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed
        }
        return new_state

    def route_based_on_evaluation(self, state: State) -> str:
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        else:
            return "worker"


    async def build_graph(self):
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)

        # Add edges
        graph_builder.add_conditional_edges("worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"})
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges("evaluator", self.route_based_on_evaluation, {"worker": "worker", "END": END})
        graph_builder.add_edge(START, "worker")

        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=self.memory)

    async def run_superstep(self, message, success_criteria, history):
        config = {"configurable": {"thread_id": self.sidekick_id}}

        state = {
            "messages": message,
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False
        }
        result = await self.graph.ainvoke(state, config=config)
        user = {"role": "user", "content": message}
        reply = {"role": "assistant", "content": result["messages"][-2].content}
        feedback = {"role": "assistant", "content": result["messages"][-1].content}
        return history + [user, reply, feedback]
    
    def cleanup(self):
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                # If no loop is running, do a direct run
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())

    async def ingest_pdf(self, pdf_path):
        print(f"Ingesting PDF: {pdf_path}")
        # Defensive: ensure RAG attributes are initialized
        if not hasattr(self, 'text_splitter') or self.text_splitter is None:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        if not hasattr(self, 'embeddings') or self.embeddings is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if not hasattr(self, 'faiss_db'):
            self.faiss_db = None
        if not hasattr(self, 'rag_docs'):
            self.rag_docs = []
        # Parse PDF
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        print(f" Extracted text from PDF (first 200 chars): {text[:200]}")
        # Split into chunks
        docs = self.text_splitter.create_documents([text])
        print(f" Number of chunks created: {len(docs)}")
        # Embed and store in FAISS
        if self.faiss_db is None:
            from langchain_community.vectorstores import FAISS
            self.faiss_db = FAISS.from_documents(docs, self.embeddings)
            print("[DEBUG] Created new FAISS DB with PDF chunks.")
        else:
            self.faiss_db.add_documents(docs)
            print("Added new chunks to existing FAISS DB.")
        self.rag_docs.extend(docs)
