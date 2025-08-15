# NeoRAG AI 

- Knowledge Graph RAG Chatbot System

A knowledge graph-based RAG (Retrieval Augmented Generation) chatbot system that combines Neo4j graph database capabilities with advanced AI models for intelligent conversation and relationship prediction.


## Project Overview

NeoRAG AI is an advanced chatbot system that leverages knowledge graphs to provide intelligent responses about complex relationships between users, analysts, contributors, issues, trends, and ideas across different platforms. The system includes sophisticated training capabilities, accuracy validation, and subject-predicate-object (SPO) relationship prediction.


## Key Features

### Core Functionality
- **Knowledge Graph Integration**: Built on Neo4j database for complex relationship modeling
- **Intelligent Retrieval**: Advanced graph traversal and pattern matching for contextual responses
- **RAG Architecture**: Combines retrieval from knowledge graph with generative AI responses
- **Multi-hop Reasoning**: Capable of understanding complex relationship chains

### Training and Validation
- **Automated Training Data Generation**: Creates diverse question-answer pairs across difficulty levels
- **Model Accuracy Validation**: Comprehensive testing with 80% accuracy target
- **SPO Triplet Learning**: Advanced subject-predicate-object relationship prediction
- **Overfitting Detection**: Monitors model performance and prevents overfitting

### Data Management
- **Synthetic Data Generation**: Creates realistic training examples when real data is limited
- **Manual Dataset Labelling**: Tools for human review and quality assurance
- **Pattern Recognition**: Learns relationship patterns for better prediction accuracy
- **Data Enhancement**: Automatically improves training datasets based on performance metrics


## System Architecture

### Components

1. **Main Chatbot Engine** (`main.py`)
   - Central orchestrator for all system components
   - User interface and interaction management
   - Integration point for all subsystems

2. **Knowledge Graph Setup** (`neo4j.py`)
   - Initial Neo4j database configuration
   - Sample data creation and population
   - Graph schema definition

3. **Triplet Management** (`triplet_supervisor.py`)
   - Subject-predicate-object triplet handling
   - Synthetic triplet generation
   - Data validation and cleanup

4. **Training Data Manager** (`training_testing_manager.py`)
   - Question-answer pair generation
   - Training and testing dataset creation
   - Model accuracy validation and recalibration

5. **SPO Training System** (`spo_training_manager.py`)
   - Advanced relationship prediction training
   - Pattern learning and recognition
   - Accuracy measurement and optimization

6. **Quality Assurance** (`random_dataset.py`)
   - Random sampling for manual review
   - Data quality assessment tools
   - Human validation workflows

7. **Configuration** (`config.py`)
   - System-wide configuration settings
   - Database connection parameters
   - Model and performance settings


## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Neo4j Database (version 4.0+)
- Ollama for local LLM inference
- Git for version control

### Required Python Packages
```
pip install neo4j
pip install sentence-transformers
pip install langchain
pip install langchain-ollama
pip install langchain-core
pip install scikit-learn
pip install numpy
pip install pydantic
```

### Neo4j Database Setup
1. Install Neo4j Desktop or Neo4j Community Server
2. Create a new database instance
3. Configure authentication credentials
4. Start the database service

### Ollama Setup
1. Install Ollama from official website
2. Pull required models:
   ```
   ollama pull llama2
   ollama pull nomic-embed-text
   ```

### Configuration
1. Update `config.py` with your database credentials:
   ```
   NEO4J_URI = "bolt://localhost:7687"      #change accordingly
   NEO4J_USER = "neo4j"  
   NEO4J_PASSWORD = "my_password"           #change accordingly
   NEO4J_DATABASE = "database_name"         #change accordingly
   ```

2. Verify Ollama model settings:
   ```
   LLM_MODEL = "llama2"
   EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
   ```


## Usage Guide

### Initial Setup
1. Run the knowledge graph initialization:
   ```
   python neo4j.py
   ```

2. Start the main system:
   ```
   python main.py
   ```

### System Capabilities

#### Option 1: Enhance Knowledge Graph
- Generates synthetic triplet data to expand the knowledge base
- Adds realistic relationships between entities
- Improves overall system knowledge coverage

#### Option 2: Create Training Data
- Generates comprehensive training and testing datasets
- Creates questions across simple, challenging, and analytical difficulty levels
- Ensures minimum 150 training samples for effective learning

#### Option 3: Validate Model Accuracy
- Tests model performance on validation datasets
- Targets 80% accuracy threshold
- Provides detailed performance breakdown by question difficulty
- Automatic recalibration when performance is below target

#### Option 4: Manual Quality Review
- Extracts random samples for human validation
- Creates structured datasets for manual labelling
- Supports quality assurance workflows

#### Option 5: SPO Relationship Training
- Advanced subject-predicate-object relationship learning
- Predicate prediction for entity pairs
- Pattern recognition and accuracy optimization
- Deterministic pattern matching for high accuracy

#### Option 6: Interactive Chat
- Conversational interface with the knowledge graph
- Intelligent response generation based on graph relationships
- Multi-hop reasoning and complex query handling

### Performance Metrics

The system monitors several key performance indicators:

- **Overall Accuracy**: Target of 80% on validation datasets
- **Question Difficulty Performance**:
  - Simple questions: Basic entity lookup and relationships
  - Challenging questions: Multi-hop reasoning and complex connections
  - Analytical questions: Pattern recognition and trend analysis
- **SPO Prediction Accuracy**: Relationship prediction between entity pairs
- **Response Quality**: Relevance and accuracy of chatbot responses

### Training Data Statistics

The system generates diverse training data:
- **Simple Questions** (60%): Direct entity queries and basic relationships
- **Challenging Questions** (30%): Multi-step reasoning and connection discovery
- **Analytical Questions** (10%): Pattern analysis and trend correlation


## Technical Implementation

### Knowledge Graph Schema
The system uses a flexible schema supporting various entity types:
- **Users**: System users who report issues and propose ideas
- **Analysts**: Domain experts who create trends and reports
- **Contributors**: Active participants who develop ideas and lead projects
- **Issues**: Problems and bugs reported in the system
- **Trends**: Analytical insights and observations
- **Ideas**: Proposed solutions and improvements
- **Platforms**: Technical systems and tools
- **Domains**: Subject matter areas and expertise fields
- **Regions**: Geographical or organizational areas
- **Impact Areas**: Functional areas affected by changes

### Relationship Types
Common relationship patterns include:
- User RAISED Issue
- Analyst AUTHORED Trend
- Contributor DEVELOPED Idea
- Issue ORIGINATED_FROM Platform
- Trend OBSERVED_IN Region
- Idea HAS_IMPACT_ON ImpactArea

### AI Models Integration
- **Embeddings**: Nomic Embed for semantic understanding
- **Language Model**: Llama2 via Ollama for response generation
- **Retrieval**: Custom Neo4j-based graph retrieval system
- **Training**: Supervised learning for relationship prediction


## File Structure

```
NeoRAG-AI/
├── main.py                          #Main system orchestrator
├── config.py                        #Configuration settings
├── neo4j.py                         #Database initialization
├── triplet_supervisor.py            #Triplet management system
├── training_testing_manager.py      #Training data and validation
├── spo_training_manager.py          #SPO relationship learning
├── random_dataset.py                #Quality assurance tools
├── README.md                        #Project documentation
└── Generated Files/
    ├── enhanced_train_dataset.json  #Training data
    ├── enhanced_test_dataset.json   #Testing data
    ├── spo_training_dataset.json    #SPO training triplets
    ├── subject_object_pairs.json    #Predicate prediction data
    ├── predicate_patterns.json      #Learned relationship patterns
    └── labelling_dataset.json       #Manual review data
```


## Advanced Features

### Accuracy Optimization
The system employs several strategies to maintain high accuracy:
- **Deterministic Pattern Matching**: Uses proven relationship patterns
- **Fallback Mechanisms**: Multiple prediction strategies for unknown patterns
- **Continuous Learning**: Adapts based on validation results
- **Error Analysis**: Identifies and addresses specific failure patterns

### Performance Optimization
- **Batch Processing**: Efficient handling of large datasets
- **Caching**: Stores frequently accessed patterns
- **Query Optimization**: Efficient Neo4j query construction
- **Parallel Processing**: Concurrent handling where possible

### Quality Assurance
- **Automated Validation**: Continuous accuracy monitoring
- **Human Review Integration**: Support for manual quality checks
- **Data Consistency**: Ensures coherent relationship patterns
- **Error Detection**: Identifies and flags problematic data


## Troubleshooting

### Common Issues

**Database Connection Problems**
- Verify Neo4j service is running
- Check credentials in config.py
- Ensure database is accessible on specified port

**Low Accuracy Scores**
- Run training data generation (Option 2)
- Execute SPO training pipeline (Option 5)
- Verify sufficient training data is available

**Slow Performance**
- Check database query optimization
- Monitor system resource usage
- Consider reducing batch sizes for processing

**Missing Dependencies**
- Install all required Python packages
- Verify Ollama installation and model availability
- Check Neo4j driver compatibility

### Performance Tuning

**For Better Accuracy**:
- Increase training data generation count
- Run multiple SPO training iterations
- Use manual quality review for data validation

**For Faster Processing**:
- Reduce validation sample sizes during development
- Optimize Neo4j query patterns
- Use batch processing for large datasets


## Contributing

The system is designed for extensibility and improvement:

1. **Adding New Entity Types**: Extend the schema in neo4j.py
2. **Custom Relationship Patterns**: Update pattern definitions in SPO training
3. **Additional Question Types**: Expand templates in training manager
4. **Performance Improvements**: Optimize queries and processing logic


## Development Roadmap

Future enhancements planned:
- **Multi-language Support**: Expand beyond English language processing
- **Advanced Analytics**: Enhanced pattern recognition and trend analysis
- **Scalability Improvements**: Support for larger knowledge graphs
- **Integration APIs**: REST and GraphQL interfaces for external integration
- **Visualization Tools**: Graph visualization and exploration interfaces


## System Requirements

### Minimum Requirements
- **CPU**: 4-core processor
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Network**: Internet connection for model downloads

### Recommended Requirements
- **CPU**: 8-core processor or better
- **RAM**: 16GB or more
- **Storage**: 50GB SSD for optimal performance
- **GPU**: Optional for enhanced AI model performance


## License and Usage

This project is designed for educational and research purposes. The system demonstrates advanced knowledge graph integration with AI models and provides a foundation for building sophisticated chatbot applications.


## Support and Documentation

For technical support and additional documentation:
- Review configuration settings in config.py
- Check system logs for error diagnostics
- Verify all dependencies are properly installed
- Ensure database connectivity and model availability


The system provides logging and error reporting to assist with troubleshooting and optimization efforts.
