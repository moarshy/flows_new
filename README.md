# flows_new

A shot at re-writing the flows repository. 

Important concepts:
1. components - basic unit. can be categorized into things like readers, vectorstore, retrievers, chat, extract_schema
2. blocks - a chain of components. a block can be serial or parallel. 
3. workflow builder and executor
