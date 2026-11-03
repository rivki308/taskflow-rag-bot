import truststore

truststore.inject_into_ssl()
import gradio as gr

from rag import ask, create_chat_engine

engine = create_chat_engine()


def respond(message, history):
    try:
        return ask(engine, message, history)
    except Exception as error:
        if "RESOURCE_EXHAUSTED" in str(error):
            return "הגעתם למגבלת השאלות לדקה בגרסה החינמית. חכו דקה ונסו שוב."
        return f"אירעה שגיאה: {error}"


demo = gr.ChatInterface(
    fn=respond,
    title="עוזר TaskFlow Pro",
    description="שאלו כל שאלה על השימוש במערכת TaskFlow Pro. העוזר עונה רק מתוך מסמכי המערכת.",
    examples=[
        "מה עובדת יכולה לעשות במערכת?",
        "מתי משימה נחשבת באיחור?",
        "איך מוסיפים עובד חדש?",
        "שכחתי סיסמה, מה עושים?",
    ],
    chatbot=gr.Chatbot(rtl=True, height=500),
    textbox=gr.Textbox(placeholder="הקלידו שאלה...", rtl=True),
)

if __name__ == "__main__":
    demo.launch()
