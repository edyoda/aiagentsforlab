from langchain_openai import ChatOpenAI     
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage                                                                                                                  
                                          
def run(message: str, session_id: str, context) -> str:
                                                                                                                                        
    # Search knowledge base                                                                                                                                                                 
    docs = context.vector_store.search(message, top_k=3)                                                                                                                                    
    rag_context = "\n".join(d["content"] for d in docs if d["similarity"] > 0.3)                                                                                                            
                                                                
    # Load conversation history                                                                                                                                                             
    history = context.memory.get_history(limit=10)            
                                                                                                                                                                                              
    system_prompt = "You are a helpful assistant."
    if rag_context:                                                                                                                                                                         
        system_prompt += f"\n\nRelevant context:\n{rag_context}"
                                              
    messages = [SystemMessage(content=system_prompt)]
    for m in history:
        cls = HumanMessage if m["role"] == "user" else AIMessage                                                                                                                            
        messages.append(cls(content=m["content"]))
    messages.append(HumanMessage(content=message))                                                                                                                                          
                                                                
      # Langfuse tracing (only if keys are configured)                                                                                                                                        
    callbacks = []                      
    if context.langfuse_secret_key and context.langfuse_public_key:                                                                                                                         
        from langfuse import Langfuse                                                                                                                                                       
        from langfuse.langchain import CallbackHandler
        Langfuse(                                                                                                                                                                           
              public_key=context.langfuse_public_key,                                                                                                                                         
              secret_key=context.langfuse_secret_key,
        )                                                                                                                                                                                   
        callbacks = [CallbackHandler()]                       
                                                                                                                                                                                              
    llm = ChatOpenAI(model="gpt-4o", api_key=context.llm_key)                                                                                                                               
    response = llm.invoke(messages, config={"callbacks": callbacks} if callbacks else {})
    return response.content  
