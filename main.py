"""
NeoRAG AI - Neo4j RAG Chatbot System
"""

import time
import numpy as np
import logging
import json
import random
import warnings
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase

#Suppress transformers warnings for clean output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers_modules").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
from config import Config

#Import core component
from triplet_supervisor import TripletSupervisor
from training_testing_manager import TrainingTestingManager
from spo_training_manager import SPOTrainingManager
from random_dataset import RandomDatasetLabeller

#Disable additional logging for clean output
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

class NomicEmbeddings(Embeddings):
    """Custom embedding wrapper for Nomic Embed model with trust_remote_code support"""
    
    def __init__(self):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL, trust_remote_code=True)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])
        return embedding.tolist()

class GraphRetriever(BaseRetriever):
    """Neo4j retriever with intelligent relationship traversal"""
    
    driver: Any = Field(description="Neo4j database driver")
    embeddings: Any = Field(description="Embeddings model")
    database: str = Field(description="Neo4j database name")
    k: int = Field(default=8, description="Number of documents to retrieve")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            with self.driver.session(database=self.database) as session:
                documents = []
                
                #Multi-layer search approach for result
                entity_docs = self._search_entity_relationships(session, query)
                documents.extend(entity_docs)
                
                link_docs = self._search_relationship_links(session, query)
                documents.extend(link_docs)
                
                path_docs = self._search_connection_paths(session, query)
                documents.extend(path_docs)
                
                unique_docs = self._remove_duplicate_documents(documents)
                return unique_docs[:self.k]
                
        except Exception:
            return []
    
    def _search_entity_relationships(self, session, query):
        """Search for specific entity relationship pattern"""
        documents = []
        
        search_patterns = [
            {
                "keywords": ["user", "issue", "platform", "status"],
                "cypher": """
                MATCH (user:User)-[:RAISED]->(issue:Issue)-[:ORIGINATED_FROM]->(platform:Platform)
                OPTIONAL MATCH (issue)-[:HAS_STATUS]->(status:Status)
                OPTIONAL MATCH (issue)-[:BELONGS_TO]->(domain:Domain)
                WHERE ANY(word IN split(toLower($search_query), ' ') 
                         WHERE toLower(user.name) CONTAINS word 
                            OR toLower(issue.name) CONTAINS word 
                            OR toLower(platform.name) CONTAINS word)
                RETURN user.name as user_name, issue.name as issue_name, 
                       platform.name as platform_name, status.name as status_name,
                       domain.name as domain_name,
                       'User ' + user.name + ' raised issue ' + issue.name + 
                       ' on platform ' + platform.name + 
                       ' with status ' + coalesce(status.name, 'Unknown') +
                       ' in domain ' + coalesce(domain.name, 'Unknown') as description
                LIMIT 3
                """,
                "type": "user_issue_platform"
            },
            {
                "keywords": ["analyst", "trend", "region", "domain"],
                "cypher": """
                MATCH (analyst:Analyst)-[:AUTHORED]->(trend:Trend)
                OPTIONAL MATCH (trend)-[:OBSERVED_IN]->(region:Region)
                OPTIONAL MATCH (trend)-[:BELONGS_TO]->(domain:Domain)
                WHERE ANY(word IN split(toLower($search_query), ' ') 
                         WHERE toLower(analyst.name) CONTAINS word 
                            OR toLower(trend.name) CONTAINS word 
                            OR toLower(region.name) CONTAINS word
                            OR toLower(domain.name) CONTAINS word)
                RETURN analyst.name as analyst_name, trend.name as trend_name,
                       region.name as region_name, domain.name as domain_name,
                       'Analyst ' + analyst.name + ' authored trend ' + trend.name +
                       ' observed in ' + coalesce(region.name, 'Unknown region') +
                       ' for domain ' + coalesce(domain.name, 'Unknown domain') as description
                LIMIT 3
                """,
                "type": "analyst_trend_region"
            },
            {
                "keywords": ["idea", "impact", "contributor", "platform"],
                "cypher": """
                MATCH (contributor:Contributor)-[:PROPOSED]->(idea:Idea)
                OPTIONAL MATCH (idea)-[:HAS_IMPACT_ON]->(impact:ImpactArea)
                OPTIONAL MATCH (idea)-[:ORIGINATED_FROM]->(platform:Platform)
                WHERE ANY(word IN split(toLower($search_query), ' ') 
                         WHERE toLower(contributor.name) CONTAINS word 
                            OR toLower(idea.name) CONTAINS word 
                            OR toLower(impact.name) CONTAINS word
                            OR toLower(platform.name) CONTAINS word)
                RETURN contributor.name as contributor_name, idea.name as idea_name,
                       impact.name as impact_name, platform.name as platform_name,
                       'Contributor ' + contributor.name + ' proposed idea ' + idea.name +
                       ' with impact on ' + coalesce(impact.name, 'Unknown area') +
                       ' from platform ' + coalesce(platform.name, 'Unknown platform') as description
                LIMIT 3
                """,
                "type": "idea_impact_analysis"
            }
        ]
        
        for pattern in search_patterns:
            if any(keyword in query.lower() for keyword in pattern["keywords"]):
                try:
                    result = session.run(pattern["cypher"], search_query=query)
                    
                    for record in result:
                        description = record.get("description", "Relationship found")
                        
                        metadata = {"type": pattern["type"]}
                        for key, value in record.items():
                            if key != "description" and value:
                                metadata[key] = value
                        
                        doc = Document(
                            page_content=description,
                            metadata=metadata
                        )
                        documents.append(doc)
                        
                except Exception:
                    continue
        
        return documents
    
    def _search_relationship_links(self, session, query):
        """Search through relationship link for connected information"""
        documents = []
        
        try:
            link_query = """
            MATCH (start_node)
            WHERE ANY(prop IN keys(start_node) 
                     WHERE toString(start_node[prop]) IS NOT NULL 
                       AND toLower(toString(start_node[prop])) CONTAINS toLower($search_term))
            
            MATCH (start_node)-[r1]->(connected1)
            OPTIONAL MATCH (connected1)-[r2]->(connected2)
            
            WITH start_node, r1, connected1, r2, connected2,
                 labels(start_node) as start_label,
                 labels(connected1) as connected1_label,
                 labels(connected2) as connected2_label,
                 coalesce(start_node.name, toString(elementId(start_node))) as start_name,
                 coalesce(connected1.name, toString(elementId(connected1))) as connected1_name,
                 coalesce(connected2.name, toString(elementId(connected2))) as connected2_name
            
            RETURN start_label, start_name, type(r1) as relationship1_type,
                   connected1_label, connected1_name, type(r2) as relationship2_type,
                   connected2_label, connected2_name
            LIMIT 5
            """
            
            result = session.run(link_query, search_term=query)
            
            for record in result:
                start_label = record["start_label"]
                start_name = record["start_name"]
                rel1_type = record["relationship1_type"]
                conn1_label = record["connected1_label"]
                conn1_name = record["connected1_name"]
                rel2_type = record.get("relationship2_type")
                conn2_label = record.get("connected2_label")
                conn2_name = record.get("connected2_name")
                
                link_description = f"{start_label} '{start_name}' -{rel1_type}-> {conn1_label} '{conn1_name}'"
                
                if rel2_type and conn2_label and conn2_name:
                    link_description += f" -{rel2_type}-> {conn2_label} '{conn2_name}'"
                
                doc = Document(
                    page_content=f"Link Analysis: {link_description}",
                    metadata={"type": "link_traversal"}
                )
                documents.append(doc)
                
        except Exception:
            pass
        
        return documents
    
    def _search_connection_paths(self, session, query):
        """Search for multi-hop connection path between entity"""
        documents = []
        
        try:
            path_query = """
            MATCH path = (start)-[*1..3]-(end)
            WHERE ANY(prop IN keys(start) 
                     WHERE toString(start[prop]) IS NOT NULL 
                       AND toLower(toString(start[prop])) CONTAINS toLower($search_term))
               OR ANY(prop IN keys(end) 
                     WHERE toString(end[prop]) IS NOT NULL 
                       AND toLower(toString(end[prop])) CONTAINS toLower($search_term))
            
            WITH path, length(path) as path_length
            WHERE path_length >= 2
            
            RETURN nodes(path) as path_nodes, 
                   relationships(path) as path_relationships,
                   path_length
            ORDER BY path_length
            LIMIT 3
            """
            
            result = session.run(path_query, search_term=query)
            
            for record in result:
                path_nodes = record["path_nodes"]
                path_relationships = record["path_relationships"]
                path_length = record["path_length"]
                
                path_description_parts = []
                for i in range(len(path_nodes)):
                    node = path_nodes[i]
                    node_label = list(node.labels) if node.labels else "Unknown"
                    node_name = node.get("name", str(node.get("id", "Unknown")))
                    
                    if i < len(path_relationships):
                        rel_type = path_relationships[i].type
                        path_description_parts.append(f"{node_label}({node_name})-[{rel_type}]->")
                    else:
                        path_description_parts.append(f"{node_label}({node_name})")
                
                path_description = "".join(path_description_parts)
                
                doc = Document(
                    page_content=f"Connection Path: {path_description}",
                    metadata={"type": "connection_path", "path_length": path_length}
                )
                documents.append(doc)
                
        except Exception:
            pass
        
        return documents
    
    def _remove_duplicate_documents(self, documents):
        """Remove duplicate document based on content similarity"""
        unique_docs = []
        seen_content = set()
        
        for doc in documents:
            content_key = doc.page_content[:100]
            if content_key not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_key)
        
        return unique_docs

class NeoRAGChatbot:
    """
    NeoRAG AI chatbot system
    """
    
    def __init__(self):
        #Initialize core Neo4j connection
        self.driver = self._create_driver()
        
        #Initialize AI model
        self.embeddings = NomicEmbeddings()
        self.llm = OllamaLLM(model=Config.LLM_MODEL)
        
        #Initialize intelligent retriever
        self.retriever = GraphRetriever(
            driver=self.driver,
            embeddings=self.embeddings,
            database=Config.NEO4J_DATABASE,
            k=Config.TOP_K_RESULTS
        )
        
        #Initialize all specialized manager
        self.triplet_supervisor = None
        self.training_manager = None
        self.spo_trainer = None
        self.dataset_labeller = None
        
        #Set up the main prompt template for reasoning
        self.prompt_template = ChatPromptTemplate.from_template("""
You are NeoRAG AI, an intelligent assistant that analyzes knowledge graph with deep understanding of relationship between users, analysts, contributors, issues, trends, and ideas across different platform.

Context from knowledge graph analysis:
{context}

Question: {question}

Answer based on the knowledge graph relationship and pattern:""")
    
    def _create_driver(self):
        """Create Neo4j database connection"""
        driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
            encrypted=False
        )
        return driver
    
    def initialize_triplet_supervisor(self):
        """Initialize the triplet management system"""
        if not self.triplet_supervisor:
            self.triplet_supervisor = TripletSupervisor(
                Config.NEO4J_URI, 
                Config.NEO4J_USER, 
                Config.NEO4J_PASSWORD, 
                Config.NEO4J_DATABASE
            )
    
    def initialize_training_manager(self):
        """Initialize the training and testing data manager"""
        if not self.training_manager:
            self.training_manager = TrainingTestingManager()
    
    def initialize_spo_trainer(self):
        """Initialize the SPO training and prediction system"""
        if not self.spo_trainer:
            self.spo_trainer = SPOTrainingManager()
    
    def initialize_dataset_labeller(self):
        """Initialize the random dataset labeller"""
        if not self.dataset_labeller:
            self.dataset_labeller = RandomDatasetLabeller()
    
    def enhance_with_synthetic_data(self, count=500):
        """Add synthetic triplet data to enhance the knowledge graph"""
        if not self.triplet_supervisor:
            self.initialize_triplet_supervisor()
        
        triplets = self.triplet_supervisor.generate_synthetic_triplets(count)
        self.triplet_supervisor.save_triplets_to_neo4j(triplets, batch_size=50)
        
        return {"status": "completed", "count": count}
    
    def create_training_testing_data(self, count=300, include_challenging=True):
        """Create training and testing dataset"""
        if not self.training_manager:
            self.initialize_training_manager()
        
        print(f"Creating training data with {count} sample (challenging data: {'Yes' if include_challenging else 'No'})...")
        
        #Generate training data with different difficulty level
        training_pairs = self.training_manager.generate_training_data(count)
        
        #Split into train and test set
        train_data, test_data = self.training_manager.create_test_dataset(training_pairs)
        
        #Save dataset to file
        self.training_manager.save_training_testing_data(train_data, test_data)
        
        return {
            "train_count": len(train_data),
            "test_count": len(test_data),
            "difficulty_distribution": {
                "simple": len([d for d in training_pairs if d.get("difficulty") == "simple"]),
                "challenging": len([d for d in training_pairs if d.get("difficulty") == "challenging"]),
                "analytical": len([d for d in training_pairs if d.get("difficulty") == "analytical"])
            }
        }
    
    def validate_accuracy_and_recalibrate(self, sample_size=80):
        """Validate model accuracy and perform automatic recalibration"""
        if not self.training_manager:
            self.initialize_training_manager()
        
        #Load test data
        try:
            with open("enhanced_test_dataset.json", "r") as f:
                test_data = json.load(f)
        except FileNotFoundError:
            print("No test data found. Please create training/testing data first.")
            return {"error": "No test data available"}
        
        print(f"Validating accuracy on {sample_size} test samples...")
        
        #Run validation
        validation_results = self.training_manager.validate_model_accuracy(self, test_data, sample_size)
        
        #Perform recalibration based on result
        calibration_info = self.training_manager.recalibrate_model(validation_results)
        
        #Get performance report
        performance_report = self.training_manager.get_performance_report()
        
        return {
            "validation_results": validation_results,
            "calibration_info": calibration_info,
            "performance_report": performance_report
        }
    
    def create_labelling_dataset(self):
        """Create random dataset for manual labelling and quality review"""
        if not self.dataset_labeller:
            self.initialize_dataset_labeller()
        
        try:
            data = self.dataset_labeller.create_labelling_dataset()
            self.dataset_labeller.save_labelling_dataset(data)
            return {
                "nodes": len(data["nodes_for_labelling"]),
                "relationships": len(data["relationships_for_labelling"])
            }
        finally:
            self.dataset_labeller.close()
    
    def run_spo_training_pipeline(self):
        """
        Run SPO training and prediction pipeline
        """
        if not self.spo_trainer:
            self.initialize_spo_trainer()
        
        print("Running SPO training pipeline...")
        
        #Step 1: Create training dataset
        self.spo_trainer.create_spo_training_dataset(min_samples=150)
        
        #Step 2: Create test pair for predicate prediction
        self.spo_trainer.create_subject_object_pairs(count=50)
        
        #Step 3: Train the predicate prediction system
        self.spo_trainer.train_predicate_prediction()
        accuracy_results = self.spo_trainer.measure_accuracy()
        
        #Step 4: Check for overfitting or underfitting
        analysis = self.spo_trainer.check_overfitting_underfitting(accuracy_results)
        
        #Step 5: Enhance training data
        if analysis["enhancement_needed"]:
            print("Enhancing training data to improve accuracy...")
            self.spo_trainer.enhance_training_data(accuracy_results, analysis)
            
            #Retrain with enhanced data
            self.spo_trainer.create_subject_object_pairs(count=50)
            self.spo_trainer.train_predicate_prediction()
            final_accuracy = self.spo_trainer.measure_accuracy()
        else:
            final_accuracy = accuracy_results
        
        #Step 6: Save SPO dataset
        self.spo_trainer.save_spo_datasets()
        
        return {
            "accuracy": final_accuracy["accuracy"],
            "meets_target": final_accuracy["meets_target"],
            "dataset_size": len(self.spo_trainer.spo_dataset),
            "test_pairs": len(self.spo_trainer.subject_object_pairs),
            "analysis": analysis,
            "target_accuracy": 0.8
        }
    
    def predict_relationship(self, subject: str, object_name: str):
        """
        Predict the relationship between subject and object
        """
        if not self.spo_trainer:
            return "Please run SPO training first"
        
        #Try to determine entity type
        subject_type = subject.split('_')[0] if '_' in subject else "Entity"
        object_type = object_name.split('_')[0] if '_' in object_name else "Entity"
        
        predicted_predicate = self.spo_trainer.predict_predicate(
            subject, subject_type, object_name, object_type
        )
        
        return f"Predicted relationship: {subject} -{predicted_predicate}-> {object_name}"
    
    def chat(self, user_query: str) -> str:
        """
        Main chat function with knowledge graph reasoning
        """
        try:
            #Retrieve relevant information from knowledge graph
            retrieved_docs = self.retriever._get_relevant_documents(user_query, run_manager=None)
            
            #Build context from retrieved document
            context_parts = []
            for doc in retrieved_docs:
                context_parts.append(doc.page_content)
            
            context = "\n".join(context_parts) if context_parts else "No specific information found in knowledge graph."
            
            #Generate response using our reasoning template
            formatted_prompt = self.prompt_template.format(context=context, question=user_query)
            response = self.llm.invoke(formatted_prompt)
            
            return response
            
        except Exception:
            return "I encountered an error processing your question. Please try rephrasing it."
    
    def close(self):
        """Clean up all connection and resource"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
        
        if self.triplet_supervisor:
            self.triplet_supervisor.close()
        
        if self.training_manager:
            self.training_manager.close()
        
        if self.spo_trainer:
            self.spo_trainer.close()
        
        if self.dataset_labeller:
            self.dataset_labeller.close()

def main():
    """
    Main function to run the complete NeoRAG AI system
    """
    chatbot = None
    
    try:
        #Initialize the NeoRAG AI system
        chatbot = NeoRAGChatbot()
        
        print("NeoRAG AI - Knowledge Graph Intelligence System")
        print("=" * 60)
        
        print("\nSystem Capabilities:")
        print("1. Enhance knowledge graph with synthetic data")
        print("2. Create training/testing dataset")
        print("3. Validate accuracy and recalibrate model") 
        print("4. Create random dataset for manual quality review")
        print("5. Run SPO triplet training and relationship prediction")
        print("6. Start intelligent chatbot")
        
        while True:
            choice = input("\nSelect option (1-6) or 'skip' to go directly to chat: ").strip()
            
            if choice == '1':
                count = int(input("How many synthetic triplets to generate? (default 300): ") or "300")
                result = chatbot.enhance_with_synthetic_data(count=count)
                print(f"Successfully added {result['count']} synthetic triplets to knowledge graph")
                
            elif choice == '2':
                count = int(input("How many training samples to create? (default 300): ") or "300")
                result = chatbot.create_training_testing_data(count=count, include_challenging=True)
                print(f"Created {result['train_count']} training and {result['test_count']} testing samples")
                print("Difficulty distribution:")
                for difficulty, sample_count in result['difficulty_distribution'].items():
                    print(f"  {difficulty}: {sample_count} samples")
                    
            elif choice == '3':
                sample_size = int(input("Validation sample size? (default 80): ") or "80")
                
                results = chatbot.validate_accuracy_and_recalibrate(sample_size=sample_size)
                if "error" not in results:
                    validation = results["validation_results"]
                    calibration = results["calibration_info"]
                    performance = results["performance_report"]
                    
                    print(f"\nValidation Results (Target: 80% accuracy):")
                    print(f"Overall accuracy: {validation['overall_accuracy']:.2%}")
                    print(f"Meets 80% target: {'YES' if validation['overall_accuracy'] >= 0.8 else 'NO'}")
                    print(f"Simple questions: {validation.get('simple_accuracy', 0):.2%}")
                    print(f"Challenging questions: {validation.get('challenging_accuracy', 0):.2%}")
                    print(f"Analytical questions: {validation.get('analytical_accuracy', 0):.2%}")
                    
                    print(f"\nModel Health Assessment:")
                    print(f"Overall health: {performance['model_health']}")
                    print(f"Needs recalibration: {calibration['needs_recalibration']}")
                    
                    if calibration["recommended_actions"]:
                        print("Recommended improvements:")
                        for action in calibration["recommended_actions"]:
                            print(f"  - {action}")
                else:
                    print(results["error"])
                    
            elif choice == '4':
                result = chatbot.create_labelling_dataset()
                print(f"Created manual labelling dataset:")
                print(f"- {result['nodes']} nodes for quality review")
                print(f"- {result['relationships']} relationships for validation")
                print("Saved to: labelling_dataset.json")
                
            elif choice == '5':
                results = chatbot.run_spo_training_pipeline()
                print(f"\nSPO Training Pipeline Results:")
                print(f"Training accuracy: {results['accuracy']:.2%}")
                print(f"Meets 80% target: {'YES' if results['meets_target'] else 'NO'}")
                print(f"Dataset size: {results['dataset_size']} triplets")
                print(f"Test pairs: {results['test_pairs']} subject-object pairs")
                print(f"Model status: {results['analysis']['status']}")
                
                if results['meets_target']:
                    print("Model is well-trained and ready for relationship prediction!")
                    
                    #Demo predicate prediction
                    demo = input("\nTry predicate prediction demo? (y/n): ").lower() == 'y'
                    if demo:
                        subject = input("Enter subject (e.g., User_1): ") or "User_1"
                        obj = input("Enter object (e.g., Issue_5): ") or "Issue_5"
                        prediction = chatbot.predict_relationship(subject, obj)
                        print(f"AI Prediction: {prediction}")
                
            elif choice == '6' or choice.lower() == 'skip':
                break
                
            else:
                print("Invalid option. Please choose 1-6 or 'skip'")
        
        print("\n" + "="*60)
        print("NeoRAG AI is ready for intelligent conversations!")
        print("Ask me about relationship, pattern, and insight from knowledge graph")
        print("="*60)
        
        # Main chat loop
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit']:
                    print("Bot: Thank you for using NeoRAG AI! Have a great day!")
                    break
                
                if not user_input:
                    continue
                
                #Get AI response
                response = chatbot.chat(user_input)
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                print("\n\nBot: Session ended. Thank you for using NeoRAG AI!")
                break
            except Exception:
                print("Bot: I'm having trouble with that question. Could you try asking it differently?")
                continue
    
    except Exception as e:
        print(f"System initialization error: {e}")
        print("Please check your configuration and try again.")
    
    finally:
        if chatbot:
            chatbot.close()
            print("NeoRAG AI system shut down successfully.")

if __name__ == "__main__":
    main()