from typing import List, Dict, Optional, Generator
from llama_cpp import Llama


class LocalModel:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 8,
                 n_gpu_layers: int = -1, verbose: bool = False):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose
        )
        self._histories: Dict[str, List[Dict[str, str]]] = {}
        self.style_profile = None
        self.personal_store = None
        self.embedder = None

    def set_style(self, profile: str):
        self.style_profile = profile

    def set_personal_memory(self, store, embedder):
        self.personal_store = store
        self.embedder = embedder

    def _build_system_prompt(self, base: str = None) -> Optional[str]:
        parts = []
        if base:
            parts.append(base)
        if self.style_profile:
            parts.append(f"Твой стиль общения:\n{self.style_profile}\nСледуй этому стилю.")
        return "\n".join(parts) if parts else None

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7,
             max_tokens: int = 512, system_prompt: Optional[str] = None) -> str:
        full_messages = []
        full_system = self._build_system_prompt(system_prompt)
        if full_system:
            full_messages.append({"role": "system", "content": full_system})
        full_messages.extend(messages)
        resp = self._llm.create_chat_completion(
            messages=full_messages, temperature=temperature, max_tokens=max_tokens
        )
        return resp["choices"][0]["message"]["content"]

    def stream_chat(self, messages, temperature=0.7, max_tokens=512,
                    system_prompt=None) -> Generator[str, None, None]:
        full_messages = []
        full_system = self._build_system_prompt(system_prompt)
        if full_system:
            full_messages.append({"role": "system", "content": full_system})
        full_messages.extend(messages)
        stream = self._llm.create_chat_completion(
            messages=full_messages, temperature=temperature,
            max_tokens=max_tokens, stream=True
        )
        for chunk in stream:
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content

    def chat_with_tools(self, messages, tools, temperature=0.7, max_tokens=512,
                        system_prompt=None):
        full_messages = []
        full_system = self._build_system_prompt(system_prompt)
        if full_system:
            full_messages.append({"role": "system", "content": full_system})
        full_messages.extend(messages)
        resp = self._llm.create_chat_completion(
            messages=full_messages, tools=tools, temperature=temperature,
            max_tokens=max_tokens
        )
        choice = resp["choices"][0]
        msg = choice["message"]
        if "tool_calls" in msg and msg["tool_calls"]:
            return {"tool_calls": msg["tool_calls"], "content": msg.get("content", "")}
        return {"content": msg.get("content", "")}

    def ask_documents(self, question, store, embedder, n_results=4,
                      temperature=0.3, max_tokens=512):
        qv = embedder.embed_query(question)
        results = store.search(qv, n_results=n_results)
        if not results["documents"] or not results["documents"][0]:
            return "В документах информация не найдена."

        parts = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            fn = meta.get("file_name", "файл")
            parts.append(f"--- Фрагмент из {fn} ---\n{doc}\n")
        ctx = "\n".join(parts)

        sp = ("Ты — эксперт-аналитик. Отвечай на основе документов. "
              "Указывай источник. Если информации нет, скажи об этом.")
        um = f"Вопрос: {question}\n\nФрагменты документов:\n\n{ctx}"
        return self.chat([{"role": "user", "content": um}], system_prompt=sp,
                         temperature=temperature, max_tokens=max_tokens)

    def chat_with_memory(self, messages, temperature=0.7, max_tokens=512,
                         system_prompt=None, n_memories=3):
        if not self.personal_store or not self.embedder:
            return self.chat(messages, temperature, max_tokens, system_prompt)
        user_msg = messages[-1]["content"] if messages else ""
        qv = self.embedder.embed_query(user_msg)
        results = self.personal_store.search(qv, n_results=n_memories)
        context_parts = []
        if results["documents"] and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                fn = meta.get("file_name", "архив")
                context_parts.append(f"--- Фрагмент из личного архива ({fn}):\n{doc}\n")
        if context_parts:
            memory_context = "Вот что ты знаешь из моих прошлых записей:\n\n" + "\n".join(context_parts)
            augmented = messages[:-1] + [{"role": "user", "content": f"{memory_context}\nВопрос: {user_msg}"}]
        else:
            augmented = messages
        return self.chat(augmented, temperature, max_tokens, system_prompt)

    def save_history(self, cid, messages):
        self._histories[cid] = messages.copy()

    def load_history(self, cid):
        return self._histories.get(cid, [])

    def clear_history(self, cid):
        if cid in self._histories:
            del self._histories[cid]
