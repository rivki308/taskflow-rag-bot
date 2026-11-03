from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import SimilarityPostprocessor

import config
from models import get_vector_store, setup_models

QA_PROMPT = PromptTemplate(
    "אתה עוזר התמיכה של מערכת TaskFlow Pro לניהול משימות בצוות.\n"
    "ענה אך ורק על סמך המידע שמופיע בין הקווים, ולא מידע כללי שאתה מכיר.\n"
    "אם התשובה לא מופיעה במידע, ענה בדיוק: \"" + config.NO_INFO_ANSWER + "\"\n"
    "ענה בשפה שבה נשאלה השאלה, בקצרה ובבהירות.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "שאלה: {query_str}\n"
    "תשובה: "
)

CONDENSE_PROMPT = PromptTemplate(
    "בהינתן היסטוריית השיחה והשאלה האחרונה של המשתמש, "
    "נסח מחדש את השאלה האחרונה כשאלה עצמאית ומלאה, שאפשר להבין בלי ההיסטוריה.\n"
    "אל תענה על השאלה, רק נסח אותה מחדש. שמור על אותה שפה.\n\n"
    "היסטוריית השיחה:\n{chat_history}\n\n"
    "השאלה האחרונה: {question}\n"
    "שאלה עצמאית: "
)


def create_chat_engine():
    setup_models()
    _, vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store)

    query_engine = index.as_query_engine(
        similarity_top_k=config.TOP_K,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=config.SIMILARITY_CUTOFF)
        ],
        text_qa_template=QA_PROMPT,
    )

    return CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        condense_question_prompt=CONDENSE_PROMPT,
    )


def to_chat_history(history):
    messages = []
    for item in history:
        content = item["content"]
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        role = MessageRole.USER if item["role"] == "user" else MessageRole.ASSISTANT
        messages.append(ChatMessage(role=role, content=str(content)))
    return messages


def ask(engine, message, history):
    response = engine.chat(message, chat_history=to_chat_history(history))

    if not response.source_nodes:
        return config.NO_INFO_ANSWER

    answer = str(response).strip()
    if answer == config.NO_INFO_ANSWER:
        return answer

    files = sorted(
        {node.metadata.get("file_name", "") for node in response.source_nodes} - {""}
    )
    if files:
        answer += "\n\n📄 מקורות: " + ", ".join(files)
    return answer
