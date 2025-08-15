"""
Triplet Supervisor for Knowledge Graph
Handle triplet operation with S-P-O separation
"""

from neo4j import GraphDatabase
from typing import List, Tuple, Dict, Any, Optional
import logging
import json
import random
import time
from datetime import datetime
from dataclasses import dataclass
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TripletData:
    """Simple structure for triplet information"""
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    metadata: Dict[str, Any]
    created_time: str

class TripletSupervisor:
    """Triplet for knowledge graph"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "testdb"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        #Keeping track
        self.stats = {
            "total_created": 0,
            "successful_saves": 0,
            "failed_saves": 0,
            "subjects_found": 0,
            "predicates_found": 0,
            "objects_found": 0
        }
        
        #Store our triplets
        self.triplet_data = []
        self.start_time = time.time()
        
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Database connection closed")
    
    def create_triplet_data(self, subject: str, predicate: str, obj: str, 
                           confidence: float = 0.8, extra_info: Dict = None) -> TripletData:
        """
        Create a new triplet with subject, predicate, and object separated
        
        Args:
            subject: The subject entity
            predicate: The relationship
            obj: The object entity
            confidence: How confident we are (0 to 1)
            extra_info: Any extra information
            
        Returns:
            TripletData object
        """
        triplet_id = f"triplet_{len(self.triplet_data) + 1}_{int(time.time())}"
        
        triplet = TripletData(
            id=triplet_id,
            subject=subject.strip(),
            predicate=predicate.strip(),
            object=obj.strip(),
            confidence=confidence,
            metadata=extra_info or {},
            created_time=datetime.now().isoformat()
        )
        
        self.triplet_data.append(triplet)
        self.stats["total_created"] += 1
        
        return triplet
    
    def generate_synthetic_triplets(self, count: int = 1000) -> List[TripletData]:
        """
        Generate some realistic triplet data
        
        Args:
            count: How many triplets to create
            
        Returns:
            List of TripletData objects
        """
        logger.info(f"Creating {count} synthetic triplets...")
        
        #Different types of subject (who/what does things)
        subject_groups = {
            "people": [f"User_{i}" for i in range(1, 51)],
            "analysts": [f"Analyst_{i}" for i in range(1, 31)],
            "contributors": [f"Contributor_{i}" for i in range(1, 31)],
            "systems": [f"System_{i}" for i in range(1, 21)],
            "companies": [f"Company_{i}" for i in range(1, 21)],
            "projects": [f"Project_{i}" for i in range(1, 41)]
        }
        
        #Different types of relationship (what they do)
        relationship_groups = {
            "actions": ["CREATED", "MODIFIED", "DELETED", "ANALYZED", "REVIEWED"],
            "connections": ["WORKS_FOR", "BELONGS_TO", "COLLABORATES_WITH", "MANAGES", "REPORTS_TO"],
            "interactions": ["RAISED", "RESOLVED", "ASSIGNED", "ESCALATED", "APPROVED"],
            "observations": ["OBSERVED_IN", "OCCURRED_AT", "DETECTED_BY", "MONITORED_BY"],
            "links": ["ASSOCIATED_WITH", "LINKED_TO", "CONNECTED_TO", "RELATED_TO"]
        }
        
        #Different types of object (what gets acted upon)
        object_groups = {
            "issues": [f"Issue_{i}" for i in range(1, 101)],
            "trends": [f"Trend_{i}" for i in range(1, 51)],
            "ideas": [f"Idea_{i}" for i in range(1, 51)],
            "platforms": ["InsightUX", "Exsight", "Afkari", "InnovateX"],
            "domains": ["Technical", "Healthcare", "Manufacturing", "Retail", "Finance", "Security"],
            "regions": ["APAC", "US", "Europe", "Global", "India", "EMEA"],
            "statuses": ["Open", "Closed", "In_Progress", "Under_Review", "Approved", "Draft"],
            "metrics": [f"Metric_{i}" for i in range(1, 31)]
        }
        
        new_triplets = []
        
        for i in range(count):
            #Pick random groups
            subject_group = random.choice(list(subject_groups.keys()))
            relationship_group = random.choice(list(relationship_groups.keys()))
            object_group = random.choice(list(object_groups.keys()))
            
            #Pick specific item from each group
            subject = random.choice(subject_groups[subject_group])
            predicate = random.choice(relationship_groups[relationship_group])
            obj = random.choice(object_groups[object_group])
            
            #Create confidence score
            confidence = random.uniform(0.6, 1.0)
            
            #Add some extra information
            extra_info = {
                "subject_type": subject_group,
                "relationship_type": relationship_group,
                "object_type": object_group,
                "source": "synthetic_data",
                "batch": f"batch_{i // 100}",
                "generator": "automatic"
            }
            
            #Create the triplet
            triplet = self.create_triplet_data(subject, predicate, obj, confidence, extra_info)
            new_triplets.append(triplet)
            
            #Update our counters
            self.stats["subjects_found"] += 1
            self.stats["predicates_found"] += 1
            self.stats["objects_found"] += 1
        
        logger.info(f"Created {len(new_triplets)} synthetic triplets")
        return new_triplets
    
    def check_triplet_quality(self, triplet: TripletData) -> Dict[str, Any]:
        """
        Check if a triplet looks good
        
        Args:
            triplet: The triplet to check
            
        Returns:
            Dictionary with quality information
        """
        quality_info = {
            "triplet_id": triplet.id,
            "looks_good": True,
            "problems": [],
            "quality_score": 1.0
        }
        
        #Check for empty part
        if not triplet.subject.strip():
            quality_info["problems"].append("Subject is empty")
            quality_info["looks_good"] = False
            quality_info["quality_score"] -= 0.3
        
        if not triplet.predicate.strip():
            quality_info["problems"].append("Predicate is empty")
            quality_info["looks_good"] = False
            quality_info["quality_score"] -= 0.3
        
        if not triplet.object.strip():
            quality_info["problems"].append("Object is empty")
            quality_info["looks_good"] = False
            quality_info["quality_score"] -= 0.3
        
        #Check confidence
        if triplet.confidence < 0.5:
            quality_info["problems"].append("Low confidence")
            quality_info["quality_score"] -= 0.1
        
        #Check length
        if len(triplet.subject) > 100:
            quality_info["problems"].append("Subject too long")
            quality_info["quality_score"] -= 0.1
        
        if len(triplet.predicate) > 50:
            quality_info["problems"].append("Predicate too long")
            quality_info["quality_score"] -= 0.1
        
        if len(triplet.object) > 100:
            quality_info["problems"].append("Object too long")
            quality_info["quality_score"] -= 0.1
        
        #Score should not go below 0
        quality_info["quality_score"] = max(0.0, quality_info["quality_score"])
        
        return quality_info
    
    def save_triplets_to_neo4j(self, triplets: List[TripletData], batch_size: int = 100):
        """
        Save triplets to Neo4j database in batches
        
        Args:
            triplets: List of triplets to save
            batch_size: How many to save at once
        """
        logger.info(f"Saving {len(triplets)} triplets to Neo4j...")
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i + batch_size]
                batch_number = i // batch_size + 1
                self._save_batch(session, batch, batch_number)
        
        logger.info(f"Finished saving triplets. Stats: {self.stats}")
    
    def _save_batch(self, session, batch: List[TripletData], batch_num: int):
        """Save one batch of triplets"""
        try:
            def save_function(tx):
                for triplet in batch:
                    #Triplet quality check
                    quality = self.check_triplet_quality(triplet)
                    
                    if quality["looks_good"]:
                        #Query to save to Neo4j
                        query = f"""
                        MERGE (s:Entity {{name: $subject}})
                        SET s.entity_role = 'subject',
                            s.last_updated = datetime()
                        
                        MERGE (o:Entity {{name: $object}})
                        SET o.entity_role = 'object',
                            o.last_updated = datetime()
                        
                        MERGE (s)-[r:`{triplet.predicate}`]->(o)
                        SET r.triplet_id = $triplet_id,
                            r.confidence = $confidence,
                            r.created_time = $created_time,
                            r.subject_type = $subject_type,
                            r.relationship_type = $relationship_type,
                            r.object_type = $object_type,
                            r.quality_score = $quality_score,
                            r.source = $source,
                            r.batch = $batch_info
                        
                        RETURN r
                        """
                        
                        result = tx.run(query, 
                                      subject=triplet.subject,
                                      object=triplet.object,
                                      triplet_id=triplet.id,
                                      confidence=triplet.confidence,
                                      created_time=triplet.created_time,
                                      subject_type=triplet.metadata.get("subject_type", ""),
                                      relationship_type=triplet.metadata.get("relationship_type", ""),
                                      object_type=triplet.metadata.get("object_type", ""),
                                      quality_score=quality["quality_score"],
                                      source=triplet.metadata.get("source", ""),
                                      batch_info=triplet.metadata.get("batch", ""))
                        
                        if result.single():
                            self.stats["successful_saves"] += 1
                    else:
                        self.stats["failed_saves"] += 1
            
            session.execute_write(save_function)
            logger.info(f"Saved batch {batch_num} with {len(batch)} triplets")
            
        except Exception as e:
            logger.error(f"Error saving batch {batch_num}: {e}")
            self.stats["failed_saves"] += len(batch)
    
    def get_existing_triplets(self, limit: int = 500) -> List[TripletData]:
        """
        Get existing relationship and turn them into triplets
        
        Args:
            limit: Maximum number to get
            
        Returns:
            List of TripletData object
        """
        logger.info(f"Getting existing triplets from database (limit: {limit})...")
        
        existing_triplets = []
        
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (s)-[r]->(o)
            WHERE s.name IS NOT NULL AND o.name IS NOT NULL
            RETURN s.name as subject, type(r) as predicate, o.name as object,
                   coalesce(r.confidence, 0.7) as confidence,
                   properties(r) as relationship_info
            LIMIT $limit
            """
            
            result = session.run(query, limit=limit)
            
            for record in result:
                extra_info = {
                    "source": "existing_database",
                    "relationship_info": dict(record["relationship_info"]),
                    "extraction_method": "database_query"
                }
                
                triplet = self.create_triplet_data(
                    subject=record["subject"],
                    predicate=record["predicate"],
                    obj=record["object"],
                    confidence=record["confidence"],
                    extra_info=extra_info
                )
                
                existing_triplets.append(triplet)
        
        logger.info(f"Got {len(existing_triplets)} existing triplets")
        return existing_triplets
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get information about what we've processed"""
        
        #Count different types
        subject_types = {}
        relationship_types = {}
        object_types = {}
        
        for triplet in self.triplet_data:
            subj_type = triplet.metadata.get("subject_type", "unknown")
            rel_type = triplet.metadata.get("relationship_type", "unknown")
            obj_type = triplet.metadata.get("object_type", "unknown")
            
            subject_types[subj_type] = subject_types.get(subj_type, 0) + 1
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        #Calculate average confidence
        if self.triplet_data:
            avg_confidence = sum(t.confidence for t in self.triplet_data) / len(self.triplet_data)
        else:
            avg_confidence = 0
        
        return {
            "processing_stats": self.stats,
            "total_triplets": len(self.triplet_data),
            "average_confidence": avg_confidence,
            "subject_types": subject_types,
            "relationship_types": relationship_types,
            "object_types": object_types,
            "processing_time": time.time() - self.start_time
        }
    
    def save_to_file(self, filename: str = "all_triplets.json"):
        """Save all triplets to a file"""
        triplet_list = []
        
        for triplet in self.triplet_data:
            triplet_list.append({
                "id": triplet.id,
                "subject": triplet.subject,
                "predicate": triplet.predicate,
                "object": triplet.object,
                "confidence": triplet.confidence,
                "metadata": triplet.metadata,
                "created_time": triplet.created_time
            })
        
        with open(filename, "w") as f:
            json.dump(triplet_list, f, indent=2)
        
        logger.info(f"Saved {len(triplet_list)} triplets to {filename}")

    #Legacy method names for backward compatibility
    def generate_synthetic_triplet_data(self, count: int = 1000) -> List[TripletData]:
        """Alias for generate_synthetic_triplets"""
        return self.generate_synthetic_triplets(count)
    
    def batch_ingest_triplets(self, triplets: List[Tuple[str, str, str]], batch_size: int = 100):
        """Legacy method - converts old triplet format to new format and saves"""
        triplet_objects = []
        for subject, predicate, obj in triplets:
            triplet_obj = self.create_triplet_data(
                subject=subject,
                predicate=predicate, 
                obj=obj,
                confidence=0.8,
                extra_info={"source": "legacy_format"}
            )
            triplet_objects.append(triplet_obj)
        
        self.save_triplets_to_neo4j(triplet_objects, batch_size)


def run_triplet_demo():
    """Show how to use the triplet supervisor"""
    #Setup
    config = {
        "uri": "neo4j://127.0.0.1:7687",
        "user": "neo4j",
        "password": "Harsh@123",
        "database": "testdb"
    }
    
    supervisor = TripletSupervisor(**config)
    
    try:
        logger.info("Starting triplet supervisor demo")
        
        #Create synthetic data
        logger.info("Creating synthetic triplets...")
        synthetic_triplets = supervisor.generate_synthetic_triplets(count=500)
        
        #Get existing data
        logger.info("Getting existing triplets...")
        existing_triplets = supervisor.get_existing_triplets(limit=200)
        
        #Combine everything
        all_triplets = synthetic_triplets + existing_triplets
        logger.info(f"Total triplets to process: {len(all_triplets)}")
        
        #Save to Neo4j
        logger.info("Saving triplets to Neo4j...")
        supervisor.save_triplets_to_neo4j(all_triplets, batch_size=50)
        
        #Get statistics
        stats = supervisor.get_statistics()
        
        #Save to file
        supervisor.save_to_file()
        
        #Show result
        print("\n" + "="*50)
        print("TRIPLET SUPERVISOR RESULTS")
        print("="*50)
        print(f"Total Triplets: {stats['total_triplets']}")
        print(f"Successful Saves: {stats['processing_stats']['successful_saves']}")
        print(f"Average Confidence: {stats['average_confidence']:.3f}")
        print(f"Failed Saves: {stats['processing_stats']['failed_saves']}")
        
        print("\nSubject Types:")
        for stype, count in stats['subject_types'].items():
            print(f"  {stype}: {count}")
        
        print("\nRelationship Types:")
        for rtype, count in stats['relationship_types'].items():
            print(f"  {rtype}: {count}")
        
        return supervisor, stats
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        return None, None
        
    finally:
        supervisor.close()


if __name__ == "__main__":
    supervisor, stats = run_triplet_demo()