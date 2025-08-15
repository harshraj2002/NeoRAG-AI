"""
Random Dataset Manager
Extract random sample from knowledge graph for manual labelling and quality assurance
"""

import random
import json
import datetime
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from config import Config

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Neo4j datetime objects and other complex types"""
    
    def default(self, obj):
        #Handle datetime objects
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        
        #Handle date objects
        if isinstance(obj, datetime.date):
            return obj.isoformat()
        
        #Handle time objects
        if isinstance(obj, datetime.time):
            return obj.isoformat()
        
        #Handle Neo4j temporal types that have isoformat method
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        
        #Handle other types that might cause issues
        if hasattr(obj, '__str__'):
            return str(obj)
        
        return super().default(obj)

class RandomDatasetLabeller:
    """
    Extract random samples from knowledge graph for quality assurance
    Support manual labelling workflow for improving data accuracy
    """
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
            encrypted=False
        )
        self.database = Config.NEO4J_DATABASE
        
        #Track sampling statistics
        self.sampling_stats = {
            "total_nodes_in_graph": 0,
            "total_relationships_in_graph": 0,
            "sampled_nodes": 0,
            "sampled_relationships": 0,
            "sampling_timestamp": None
        }
    
    def get_graph_statistics(self):
        """Get basic statistics about the knowledge graph"""
        with self.driver.session(database=self.database) as session:
            try:
                #Count total node
                node_count_result = session.run("MATCH (n) RETURN count(n) as total_nodes")
                total_nodes = node_count_result.single()["total_nodes"]
                
                #Count total relationship
                rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) as total_rels")
                total_rels = rel_count_result.single()["total_rels"]
                
                self.sampling_stats["total_nodes_in_graph"] = total_nodes
                self.sampling_stats["total_relationships_in_graph"] = total_rels
                
                return {
                    "total_nodes": total_nodes,
                    "total_relationships": total_rels
                }
                
            except Exception as e:
                print(f"Error getting graph statistics: {e}")
                return {"total_nodes": 0, "total_relationships": 0}
    
    def get_random_nodes(self, count=30):
        """
        Extract random node from knowledge graph
        Uses corrected Cypher syntax for counting relationship
        """
        with self.driver.session(database=self.database) as session:
            #Cypher query for deprecation
            query = """
            MATCH (n)
            WHERE n.name IS NOT NULL
            OPTIONAL MATCH (n)-[out]->()
            OPTIONAL MATCH (n)<-[inc]-()
            WITH n, count(DISTINCT out) as outgoing_relationships, count(DISTINCT inc) as incoming_relationships
            RETURN elementId(n) as node_id, 
                   labels(n) as node_type, 
                   n.name as name, 
                   properties(n) as props, 
                   outgoing_relationships, 
                   incoming_relationships
            """
            
            result = session.run(query)
            all_nodes = []
            
            for record in result:
                outgoing = record.get("outgoing_relationships", 0)
                incoming = record.get("incoming_relationships", 0)
                
                node_data = {
                    "node_id": record.get("node_id"),
                    "node_type": record.get("node_type"),
                    "name": record.get("name"),
                    "props": self._serialize_properties(record.get("props", {})),
                    "outgoing_relationships": outgoing,
                    "incoming_relationships": incoming,
                    "total_connections": outgoing + incoming
                }
                all_nodes.append(node_data)
            
            #Sort by connection strength for better sampling
            all_nodes.sort(key=lambda x: x["total_connections"], reverse=True)
            
            #Select diverse sample of node
            if len(all_nodes) <= count:
                random_nodes = all_nodes
            else:
                #Mix high-connected and random node
                top_nodes = all_nodes[:count//2]
                remaining_nodes = all_nodes[count//2:]
                random.shuffle(remaining_nodes)
                additional_nodes = remaining_nodes[:count - len(top_nodes)]
                random_nodes = top_nodes + additional_nodes
                random.shuffle(random_nodes)
            
            self.sampling_stats["sampled_nodes"] = len(random_nodes)
            return random_nodes
    
    def get_entity_relationships(self, node_ids, rel_count=50):
        """
        Get relationship involving the selected node
        Includes relationship metadata for quality assessment
        """
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (a)-[r]->(b)
            WHERE elementId(a) IN $node_ids OR elementId(b) IN $node_ids
            RETURN elementId(a) as source_id, 
                   a.name as source_name, 
                   labels(a) as source_type,
                   type(r) as relationship_type,
                   properties(r) as relationship_props,
                   elementId(b) as target_id, 
                   b.name as target_name, 
                   labels(b) as target_type,
                   elementId(r) as relationship_id
            LIMIT $rel_count
            """
            
            result = session.run(query, node_ids=node_ids, rel_count=rel_count)
            relationships = []
            
            for record in result:
                rel_data = {
                    "relationship_id": record.get("relationship_id"),
                    "source_id": record.get("source_id"),
                    "source_name": record.get("source_name"),
                    "source_type": record.get("source_type"),
                    "relationship_type": record.get("relationship_type"),
                    "relationship_props": self._serialize_properties(record.get("relationship_props", {})),
                    "target_id": record.get("target_id"),
                    "target_name": record.get("target_name"),
                    "target_type": record.get("target_type"),
                }
                relationships.append(rel_data)
            
            self.sampling_stats["sampled_relationships"] = len(relationships)
            return relationships
    
    def _serialize_properties(self, props):
        """
        Convert Neo4j properties to JSON-serializable format
        Handle datetime objects and other complex Neo4j type
        """
        if not props:
            return {}
        
        serialized = {}
        for key, value in props.items():
            try:
                if isinstance(value, datetime.datetime):
                    serialized[key] = value.isoformat()
                elif isinstance(value, datetime.date):
                    serialized[key] = value.isoformat()
                elif isinstance(value, datetime.time):
                    serialized[key] = value.isoformat()
                elif hasattr(value, 'isoformat'):   #Neo4j temporal type
                    serialized[key] = value.isoformat()
                elif isinstance(value, (list, tuple)):
                    #Handle list/array
                    serialized[key] = [str(item) for item in value]
                else:
                    serialized[key] = value
            except Exception:
                #Fallback to string representation
                serialized[key] = str(value)
        
        return serialized
    
    def create_labelling_dataset(self):
        """
        Create dataset for manual labelling and quality review
        Provide structured format for human reviewer
        """
        print("Extracting random dataset from knowledge graph...")
        
        #Get graph statistics first
        graph_stats = self.get_graph_statistics()
        
        #Get random node (default 30)
        random_nodes = self.get_random_nodes(30)
        node_ids = [node['node_id'] for node in random_nodes]
        
        #Get relationship involving these node (default 50)
        relationships = self.get_entity_relationships(node_ids, 50)
        
        #Update sampling timestamp
        self.sampling_stats["sampling_timestamp"] = datetime.datetime.now().isoformat()
        
        #Create comprehensive labelling structure
        labelling_data = {
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "created_for": "manual_labelling_and_quality_review",
                "graph_statistics": graph_stats,
                "sampling_statistics": self.sampling_stats,
                "instructions": {
                    "node_labelling": "Review and correct node names, types, and add quality scores",
                    "relationship_validation": "Verify relationship correctness and suggest improvements",
                    "quality_scoring": "Rate confidence on scale of 1-5 (5=highest confidence)"
                }
            },
            "nodes_for_labelling": [],
            "relationships_for_labelling": []
        }
        
        #Format node for manual review
        for node in random_nodes:
            node_for_review = {
                "node_id": node['node_id'],
                "current_name": node['name'],
                "current_type": node['node_type'],
                "current_properties": node['props'],
                "connection_strength": {
                    "outgoing": node['outgoing_relationships'],
                    "incoming": node['incoming_relationships'],
                    "total": node['total_connections']
                },
                
                #Fields for manual input
                "quality_assessment": {
                    "name_quality": "",  #Rate 1-5
                    "type_accuracy": "",  #Rate 1-5  
                    "overall_quality": ""  #Rate 1-5
                },
                "corrections": {
                    "suggested_name": "",  #Corrected name if needed
                    "suggested_type": "",  #Corrected type if needed
                    "additional_properties": ""  #Suggest additional property
                },
                "reviewer_notes": "",  #General notes from reviewer
                "review_status": ""  #pending/reviewed/approved
            }
            labelling_data["nodes_for_labelling"].append(node_for_review)
        
        #Format relationship for manual validation
        for rel in relationships:
            rel_for_review = {
                "relationship_id": rel['relationship_id'],
                "relationship_triple": {
                    "source": f"{rel['source_type']}:{rel['source_name']}",
                    "relationship": rel['relationship_type'],
                    "target": f"{rel['target_type']}:{rel['target_name']}"
                },
                "relationship_properties": rel['relationship_props'],
                
                #Fields for manual validation
                "validation": {
                    "is_semantically_correct": "",  #true/false
                    "confidence_score": "",  #1-5 scale
                    "relationship_strength": ""  #weak/medium/strong
                },
                "corrections": {
                    "suggested_relationship_type": "",  #If current is wrong
                    "suggested_direction": "",  #Should it be reversed?
                    "additional_context": ""   #Missing context or property
                },
                "reviewer_notes": "",  #Specific note about this relationship
                "review_status": ""  #pending/reviewed/approved/rejected
            }
            labelling_data["relationships_for_labelling"].append(rel_for_review)
        
        return labelling_data
    
    def save_labelling_dataset(self, data):
        """
        Save the labelling dataset to JSON file with proper encoding
        Create a file ready for human reviewer
        """
        try:
            with open("labelling_dataset.json", "w") as f:
                json.dump(data, f, indent=2, cls=DateTimeEncoder)
            
            print(f"Manual labelling dataset saved successfully:")
            print(f"- {len(data['nodes_for_labelling'])} nodes for quality review")
            print(f"- {len(data['relationships_for_labelling'])} relationships for validation")
            print(f"- Saved to: labelling_dataset.json")
            print(f"- Total graph nodes: {data['metadata']['graph_statistics']['total_nodes']}")
            print(f"- Total graph relationships: {data['metadata']['graph_statistics']['total_relationships']}")
            
        except Exception as e:
            print(f"Error saving labelling dataset: {e}")
    
    def load_reviewed_dataset(self, filename="reviewed_dataset.json"):
        """
        Load manually reviewed dataset
        Processes human feedback for quality improvement
        """
        try:
            with open(filename, "r") as f:
                reviewed_data = json.load(f)
            
            print(f"Loaded reviewed dataset from {filename}")
            return reviewed_data
            
        except FileNotFoundError:
            print(f"Reviewed dataset file {filename} not found")
            return None
        except Exception as e:
            print(f"Error loading reviewed dataset: {e}")
            return None
    
    def apply_manual_corrections(self, reviewed_data):
        """
        Apply manual correction back to the knowledge graph
        Update the graph based on human review feedback
        """
        if not reviewed_data:
            print("No reviewed data to apply")
            return
        
        print("Applying manual correction to knowledge graph...")
        
        with self.driver.session(database=self.database) as session:
            corrections_applied = 0
            
            #Apply node correction
            for node_review in reviewed_data.get("nodes_for_labelling", []):
                if node_review.get("review_status") == "approved":
                    corrections = node_review.get("corrections", {})
                    
                    if corrections.get("suggested_name") or corrections.get("suggested_type"):
                        try:
                            #Update node with correction
                            update_query = """
                            MATCH (n)
                            WHERE elementId(n) = $node_id
                            SET n.reviewed_name = $suggested_name,
                                n.reviewed_type = $suggested_type,
                                n.quality_score = $quality_score,
                                n.manual_review_completed = true,
                                n.review_timestamp = datetime()
                            """
                            
                            session.run(update_query,
                                       node_id=node_review["node_id"],
                                       suggested_name=corrections.get("suggested_name", ""),
                                       suggested_type=corrections.get("suggested_type", ""),
                                       quality_score=node_review.get("quality_assessment", {}).get("overall_quality", 0))
                            corrections_applied += 1
                            
                        except Exception as e:
                            print(f"Error updating node {node_review.get('current_name')}: {e}")
            
            #Apply relationship correction
            for rel_review in reviewed_data.get("relationships_for_labelling", []):
                if rel_review.get("review_status") == "approved":
                    corrections = rel_review.get("corrections", {})
                    validation = rel_review.get("validation", {})
                    
                    if validation.get("is_semantically_correct") == "true":
                        try:
                            #Mark relationship as validated
                            update_query = """
                            MATCH ()-[r]->()
                            WHERE elementId(r) = $rel_id
                            SET r.confidence_score = $confidence,
                                r.relationship_strength = $strength,
                                r.human_validated = true,
                                r.validation_timestamp = datetime()
                            """
                            
                            session.run(update_query,
                                       rel_id=rel_review["relationship_id"],
                                       confidence=validation.get("confidence_score", 0),
                                       strength=validation.get("relationship_strength", ""))
                            corrections_applied += 1
                            
                        except Exception as e:
                            print(f"Error updating relationship: {e}")
            
            print(f"Applied {corrections_applied} manual corrections to knowledge graph")
            return corrections_applied
    
    def close(self):
        """Clean up database connection"""
        if self.driver:
            self.driver.close()

def create_manual_labelling_workflow():
    """
    Complete workflow for manual labelling and quality assurance
    This function demonstrate the full process from data extraction to application
    """
    print("Starting Manual Labelling Workflow for NeoRAG AI")
    print("=" * 50)
    
    labeller = RandomDatasetLabeller()
    
    try:
        #Step 1: Create labelling dataset
        print("Step 1: Creating labelling dataset...")
        data = labeller.create_labelling_dataset()
        
        #Step 2: Save dataset for manual review
        print("Step 2: Saving dataset for manual review...")
        labeller.save_labelling_dataset(data)
        
        #Step 3: Show preview of what need to be reviewed
        print("\nStep 3: Preview of data for manual review...")
        print("\nSample node for quality assessment:")
        for i, node in enumerate(data["nodes_for_labelling"][:3]):
            print(f"{i+1}. {node['current_type']}: {node['current_name']} (Connections: {node['connection_strength']['total']})")
        
        print("\nSample relationship for validation:")
        for i, rel in enumerate(data["relationships_for_labelling"][:3]):
            triple = rel["relationship_triple"]
            print(f"{i+1}. {triple['source']} -{triple['relationship']}-> {triple['target']}")
        
        print(f"\nNext step for manual review:")
        print("1. Open labelling_dataset.json")
        print("2. Fill in quality_assessment and correction field")
        print("3. Save as reviewed_dataset.json")
        print("4. Run apply_manual_corrections() to update the graph")
        
        return data
        
    except Exception as e:
        print(f"Error in manual labelling workflow: {e}")
        return None
        
    finally:
        labeller.close()

if __name__ == "__main__":
    #Run the manual labelling workflow
    dataset = create_manual_labelling_workflow()
    
    if dataset:
        print("\n" + "="*50)
        print("Manual labelling dataset created successfully!")
        print("Ready for human quality review and validation.")
        print("="*50)