Here is a todo list to work on for the next agent, formed by my thoughts on 4/23/2026

1. There is an obviously exploitable behavior in the mixed conditional model, it still makes completely nonsensical bids when the handsize would otherwise forbid that hand. I think likely what is happening is that the mixed strategy when not implementing its hard coded bid will bid a random hand. This leads to flushes and 4 of a kinds when less than 5 cards are possible. It should be basically impossible to bid these hands even with only 10 cards on the table. 
    1.We need to brainstorm fixes to this, I think a promising start is to implement 1/4 of the time bidding the conditionally highest hand, 1/4 of the time bidding the minimum viable hand, 1/4 bidding a hand in between that and 1/4 of the time bidding a slightly higher hand than this. 
    2. Perhaps adjusting these weights dynamically. Also if an opponent ever makes an impossible bid it should always call bluff. 
    3. In this way, the bidding space is actually highly limited, at any given time there are less than 10 or viable bids. 
    4. likewise the response needs to be dynamic, if an opponent over bids a hand that was already at the upper end of the viable hands, then it needs to call the bluff. 
    5. The fundamental idea of the benchmark ladder of agents building off the conditional probabilities is good, but this needs to be reworked according to some of these ideas. 
    6. To this end, a good agent should have some fail safe to ensure it never makes drastically unviable bids. 
2. The files in the directory have grown bloated and the number of markdown files is immense. A more standardized training and file structure system needs to be implemented, so that working on the agents and training the agents can be properly separated and context is not wasted on reading unneeded files. 
    1. This will likely take a lot of work and need a full session or more. 
    2. As much of the process of benchmarking and training agents should be as automated as possible, including looking at individual game logs and assessing the flaws a particular agent will have and how they can be fixed. This sort of reflection process should be automated as well and will take time to develop.
    3. Context and tokens are limited resource and more choices should be made to conserve these. Brainstorm ideas of how to streamline this process according to best development practices. 
3. In general going forward, new best practices will need to be implemented before making implementations for major features, the following must be done: 
    1. The model should create a design doc with the requirements, specifications, and I/O of the feature, asking relevant questions as needed. 
    2. Then the model should make a plan/checklist doc of everything that will be needed for that feature. 
    3. Then the model should write a testing checklist of all functionalities to test including edge cases.
    4. After the above 3 things have been completed, implementation can start.  