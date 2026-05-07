import anthropic
from langfuse import Langfuse 

def run(message: str, session_id: str, context) -> str:

    print(f"[langfuse] secret_key present: {bool(context.langfuse_secret_key)}")                                                                                                            
    print(f"[langfuse] public_key: {context.langfuse_public_key}")


    lf = Langfuse(                                
        secret_key=context.langfuse_secret_key,                                                                                                                                             
        public_key=context.langfuse_public_key,           
        host="https://us.cloud.langfuse.com",
        ) 
    try:  
                                                                                                                                                                                            
        lf.auth_check()
        print("[langfuse] auth_check PASSED")                                                                                                                                                   
    except Exception as e:                                    
        print(f"[langfuse] auth_check FAILED: {e}")

    # Search knowledge base
    docs = context.vector_store.search(message, top_k=3)
    rag_context = "\n".join(d["content"] for d in docs if d["similarity"] > 0.3)

    # Load conversation history
    history = context.memory.get_history(limit=10)

    system_prompt = "You are a helpful assistant."
    if rag_context:
        system_prompt += f"\n\nRelevant context:\n{rag_context}"

    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    print(f"[anthropic] {len(docs)} RAG docs, {len(history)} history turns")
    trace = lf.trace(name="agent-run", session_id=session_id, input=message)
    print(f"[langfuse] trace created : {trace.id}")

    client = anthropic.Anthropic(api_key=context.llm_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )
    result = response.content[0].text                     
                                       
    trace.update(output=result)    
    print("[langfuse] calling shutdown ...")                                                                                                                                                      
    lf.shutdown()                          
    print("[langfuse] shutdown complete")    
    return result
