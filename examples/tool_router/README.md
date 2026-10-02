# Real Use Case: Tool Router

Neo can sit before an agent as a low-latency tool router. The agent supplies
the current message and the tools currently available. Neo chooses one tool or
`none`, returns probabilities, and the application applies its own threshold:

- probability >= 0.90: route automatically;
- 0.60–0.90: send to a stronger model or ask for clarification;
- below 0.60: abstain and request human review.

This example is a routing layer, not an autonomous permission system. The
application must still validate tool arguments and authorization.

Canonical model reference: https://huggingface.co/tokenaii/Neo
