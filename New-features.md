Here is a todo list to work on for the next agent, formed by my thoughts on 4/23/2026

1. The files in the directory have grown bloated and the number of markdown files is immense. A more standardized training and file structure system needs to be implemented, so that working on the agents and training the agents can be properly separated and context is not wasted on reading unneeded files. 
    1. This will likely take a lot of work and need a full session or more. 
    2. As much of the process of benchmarking and training agents should be as automated as possible, including looking at individual game logs and assessing the flaws a particular agent will have and how they can be fixed. This sort of reflection process should be automated as well and will take time to develop.
    3. Context and tokens are limited resource and more choices should be made to conserve these. Brainstorm ideas of how to streamline this process according to best development practices. 
2. In general going forward, new best practices will need to be implemented before making implementations for major features, the following must be done: 
    1. The model should create a design doc with the requirements, specifications, and I/O of the feature, asking relevant questions as needed. 
    2. Then the model should make a plan/checklist doc of everything that will be needed for that feature. 
    3. Then the model should write a testing checklist of all functionalities to test including edge cases.
    4. After the above 3 things have been completed, implementation can start.  
3. Some current problems that should be fixed in the new design agent are as follows:
    1. Changing the agents client side as well as web side is wasteful, requiring double the amount of edits. Either a new hosting set up or simpler front end integration is needed. 
    2. When benchmarking agents, not as many tests are needed, it is more important to keep track of the choices made during games so looking at that data can be used to automatically highlight potential errors and room for improvement. This will require new I/O 
    3. The file system needs to be organized for future growth better. 
    4. The current Nash and CFR agents are boderline useless, they will have to be fully redesigned for higher hand sizes from the start so that they are not obsolete. 
    5. More advanced knowledge from start of the art poker training AI techniques will have to be used, potentially worked on in parts and a literature search will be needed to make sure that the techniques used are state of the art. 