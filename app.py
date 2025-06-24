import os
from dotenv import load_dotenv
load_dotenv(override=True)

import gradio as gr
from agentflow_tools import create_calendar_event, list_upcoming_events, ingest_pdf_for_rag
from agentflow import AgentFlow
from gradio import update

async def setup():
    agentflow = AgentFlow()
    await agentflow.setup()
    return agentflow

async def process_message(agentflow, message, success_criteria, history):
    results = await agentflow.run_superstep(message, success_criteria, history)
    return results, agentflow

async def reset():
    new_agentflow = AgentFlow()
    await new_agentflow.setup()
    return "", "", None, new_agentflow


def free_resources(agentflow):
    print("Cleaning up")
    try:
        if agentflow:
            agentflow.cleanup()
    except Exception as e:
        print(f"Exception during cleanup: {e}")

# Gradio UI
with gr.Blocks(title="AgentFlow – Multi-Agent Workflow Orchestration Assistant", theme=gr.themes.Default(primary_hue="emerald")) as ui:
    gr.Markdown("## AgentFlow – Multi-Agent Workflow Orchestration Assistant")
    agentflow = gr.State(delete_callback=free_resources)

    with gr.Row():
        chatbot = gr.Chatbot(label="AgentFlow", height=300, type="messages")
    with gr.Group():
        with gr.Row():
            message = gr.Textbox(show_label=False, placeholder="Your request to AgentFlow")
        with gr.Row():
            success_criteria = gr.Textbox(show_label=False, placeholder="What are your success criteria?")
        with gr.Row():
            pdf_file = gr.File(label="Upload PDF for RAG", file_types=[".pdf"])
            ingest_status = gr.Markdown("", visible=False)
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")

    # Bind main functions
    ui.load(setup, [], [agentflow])
    message.submit(process_message, [agentflow, message, success_criteria, chatbot], [chatbot, agentflow])
    success_criteria.submit(process_message, [agentflow, message, success_criteria, chatbot], [chatbot, agentflow])
    go_button.click(process_message, [agentflow, message, success_criteria, chatbot], [chatbot, agentflow])
    reset_button.click(reset, [], [message, success_criteria, chatbot, agentflow])

    # Add PDF ingestion logic (optional, does not interfere with chat)
    async def handle_pdf_upload(agentflow, pdf_file):
        if pdf_file is None:
            return update(value="No PDF uploaded.", visible=True)
        try:
            await agentflow.ingest_pdf(pdf_file.name)
            return update(value="PDF ingested for RAG.", visible=False)
        except Exception as e:
            return update(value=f"Error ingesting PDF: {e}", visible=True)
    pdf_file.upload(handle_pdf_upload, [agentflow, pdf_file], [ingest_status])

ui.launch(inbrowser=True)