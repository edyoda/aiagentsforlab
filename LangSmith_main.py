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
                                                                                                                                                                                              
      # LangSmith tracing
      if context.langsmith_api_key:                                                                                                                                                           
          import os                                             
          os.environ["LANGCHAIN_TRACING_V2"] = "true"                                                                                                                                         
          os.environ["LANGCHAIN_API_KEY"] = context.langsmith_api_key
          os.environ["LANGCHAIN_PROJECT"] = "edyoda-agents"                                                                                                                                   
                                              
      llm = ChatOpenAI(model="gpt-4o", api_key=context.llm_key)                                                                                                                               
      response = llm.invoke(messages)                           
      return response.content                                     
