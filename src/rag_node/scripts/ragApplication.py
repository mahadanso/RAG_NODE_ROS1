#!/usr/bin/env python

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sys
from pathlib import Path

parent_dir = Path(__file__).parent
parent_dir = parent_dir
print(f"Parent directory being added to sys.path: {parent_dir}")
sys.path.append(str(parent_dir))  # Ensure parent directory is in sys.path

import rospy
from rag_node.srv import Prompt, PromptResponse, CreateCollection, CreateCollectionResponse
from ragImplementation import *

config = None
conversation_history = None
collection = None

def handle_prompt_request(req):
    """
    Callback function to handle prompt service requests.
    """
        
    global conversation_history, collection, config
    
    verbose_mode = config.get('verboseMode', 'false').lower() == 'true'
    
    if verbose_mode:
        rospy.loginfo("Received prompt request: %s", req.prompt)

    ai_response = handle_rag_query(collection, req.prompt, conversation_history, verbose_mode, int(config.get('topK', 3)), )

    conversation_history.append({"role": "user", "content": req.prompt, "response": ai_response})

    response = PromptResponse()
    
    # Keep conversation history manageable
    if len(conversation_history) > 5:
        conversation_history = conversation_history[-3:]

    if verbose_mode:
        rospy.loginfo("AI response: %s", ai_response)

    response.response = ai_response
    return response

def handle_prompt_intent_request(req):
    """
    Callback function to handle prompt intent service requests.
    """
        
    global conversation_history, collection, config
    
    verbose_mode = config.get('verboseMode', 'false').lower() == 'true'
    
    if verbose_mode:
        rospy.loginfo("Received prompt request: %s", req.prompt)

    ai_response = handle_rag_query(collection, req.prompt, conversation_history, verbose_mode, int(config.get('topK', 3)), intent=True)

    conversation_history.append({"role": "user", "content": req.prompt, "response": ai_response})

    response = PromptResponse()
    
    # Keep conversation history manageable
    if len(conversation_history) > 5:
        conversation_history = conversation_history[-3:]

    if verbose_mode:
        rospy.loginfo("AI response: %s", ai_response)

    response.response = ai_response
    return response

def handle_create_collection_request(req):
    """
    Callback function to handle create collection service requests.
    """
        
    global collection, config
    
    verbose_mode = config.get('verboseMode', 'false').lower() == 'true'
    
    if verbose_mode:
        rospy.loginfo("Received create collection request: %s, %s, %s", req.name, req.datafile_path, req.description)
        
    global collection
    collection = create_collection_and_load_data(req.name, req.description, req.datafile_path, verbose_mode)

    response = CreateCollectionResponse()
    
    if collection:
        response.success = 1
        response.message = f"Collection '{req.name}' created successfully."

        if verbose_mode:
            rospy.loginfo(response.message)
    else:
        response.success = 0
        response.message = f"Failed to create collection '{req.name}'. Debug logs for details."
        if verbose_mode:
            rospy.loginfo(response.message)
                
    
    return response

def rag_service_server():
    """
    Initializes the ROS node and advertises the service.
    """
    rospy.init_node('rag_service_server_node')
    s = rospy.Service('rag_service/prompt', Prompt, handle_prompt_request)
    s2 = rospy.Service('rag_service/create_collection', CreateCollection, handle_create_collection_request)
    s3 = rospy.Service('rag_service/prompt_intent', Prompt, handle_prompt_intent_request)

    rospy.loginfo("RAG service server ready.")
    
    # print(f"Parent directory before reading config: {Path(__file__).parent.parent}")

    global config
    config = read_config(Path(__file__).parent.parent / 'config' / 'ragSystem.ini')

    # Get collection for RAG system
    global collection
    collection = get_similarity_search_collection("interactive_upanzi_search")

    if collection is None:
        rospy.logerr("Call the create_collection service to create a collection before using the RAG system.")

    global conversation_history
    conversation_history = []

    rospy.spin() # Keep the node alive until shutdown

if __name__ == "__main__":
    rag_service_server()