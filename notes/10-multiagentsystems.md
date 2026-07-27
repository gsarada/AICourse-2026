## Types of Agents
1. Function-calling agents - Bridge the gap between language understanding and coordinated sequence actions
2. Memory-augmented agents - They maintain complex state across interactions, update knowledge with new information and learn from past mistakes
3. Specialist agents - Do a particular task more focused and reliably
    - Research agent - search and synthesise information
    - Validation agent - ensure accuracy and consistency
    - Planning agent - break down problem into small tasks
    - Execution agent - carry out specific actions

## Structuring multi-agent interactions
1. Hierarchical structure - a planning agent coordinates multiple specialized agents
2. Peer network - agents collaborating freely, sharing capabilities

## Multi agent communication protocols
1. Function call structure - Defined with a fixed schema containing function name, description specifying 
   what a function can do and when it can be called, and parameters schema defining what properties are accepted by the function

## Context Management
1. Immediate context - information that has been provided in current turn
2. Historic context - context from historical conversation, stored preferences
3. System context - Real time data retrieved from external sources/environment

## Error handling
1. Input validation
2. API failures
3. Intelligent retry strategies
4. Fallback strategies

## Coordination strategies
1. Chain of Thought with Function calling. 
   - Ex: Understand request, plan approach, identify needed functions, execute and verify, refine based on results
2. Validation and Verification
   - Ex: validation for security, verification for performance, validation for consistency

## LLMs as connected agents
1. LLMs can use Plugin architecture to learn about system. The architecture components include 
   - Manifest: defines plugin capabilities (example in case of API, define endpoint, required inputs, accepted formats, expected API response)
   - Authentication Layer: manages secure access to resources (manage API keys, implement rate limiting to avoid abuse)
   - Response handling: ensure LLM can interpret and use results effectively (standardise API response formats, 
     handle errors gracefully, convert units, formats dates and aggregates data for LLM use)
2. Considerations when interacting with real world systems    
   - Reference of what an SQL query generation agent should go through - Understand intent, schema mapping
    query optimization, security considerations, safety checks (validation of generated queries, parameter sanitization, access control enforcement, proper escaping of query parameters)
   - Reference of what an API interaction agent should go through - API quota, implement backoff strategies, prepare headers, service availability, handling invalid input errors, 
    handle unexpected response format, maintain context across multiple API calls, cache response when appropriate, handle session mgmt
3. Integration patterns
   - Direct Integration. It is quick, simple setup. Limited by lack of caching, minimal error handling and poses security risks
   - Proxy Services. Adds a layer between LLM and external APIs. Handles rate limiting, caches common requests, provides better error handling,
     adds security controls and monitoring
   - Event Driven. Handles async operations, offers scalability
## Security Considerations
1. Access Control
   - API Keys: Define identifiers to authenticate agents access to external services
   - Authentication: Verify agents identity before granting access
   - Permission levels: Define ganular control on what agents can perform
   - Context-aware access: Permissions depending on workflow or situational relevance
2. Data Protection
   - Input validation: Ensure input either from users or agents is well-formed and safe
   - Output sanitisation: Filter agent outputs to remove or mask sensitive content
   - Sensitive Data Handling: Limit access and use encryption or redaction when logging or transmitting data
3. Function Safety: Clearly define what agents can do preventing unintended operations
   - Allowed Operations: Restrict to specific tasks such as read-only
   - Resource limits: Set thresholds for memory, CPU usage and API quota
   - Execution boundaries: Define sandbox environments to contain actions safely
4. Monitoring and Audit Trails
   - Access Logs: Record who accesses what, when and why
   - Usage patterns: Track behavioural trends across agents to identify anomalies
   - Alert Systems: Trigger notifications based on threshold breaches or unusual activity
   - Sensitive logging: Encrypt or redact sensitive data when logging
